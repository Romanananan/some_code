import pickle, time, joblib, pyspark, pandas as pd, re, gc,\
        os, numpy as np, sh, sys, datetime as dt
from typing import Any
from pyspark.sql import functions as F
from pyspark.sql import types as T


class Dm_building:

    def __init__(self,
                 spark: pyspark.sql.session.SparkSession,
                 dm_path: str,
                 hdfs_path: str,
                 dm_folder: str,
                 dm_name: str,
                 slice_date: str,
                 numeric_id_name: str,
                 print_func: Any,
                 ch_parq_table_func: Any,
                 transformer_path: str,
                 hive_query_func: Any,
                 features_from_dm: list,
                 features_for_avg_aggr: list,
                 score_on_ready_data_mart: bool = False,
                 features_to_transform: list or bool = False,
                 dm_periods_column_name: str = 'report_dt_part',
                 slice_date_pattern: str = '%Y-%m-%d',
                 periods_from_dm: list or tuple = (3, 6, 12),
                 own_module: str = 'dspl',
                 max_batch_size_for_encoding: int = int(2e7),
                 batch_size: int = int(7e5),
                 create_tables_in_hive: bool or str = False,
                 hive_database_for_dms='default',
                 dm_filter_cond: str or bool = f'sd_client_valid_nflag = 1 and sd_dead_nflag = 0'
                 ):
        """
        create_tables_in_hive - False or name of database for creating external table from parquets
        dm_periods_column_name - name of column in dm (it could be better if it's main partition col in your dm)
        dm_path - full path in hadoop to dm
        numeric_id_name - name of column with uniq client id
        hdfs_path - full path in hadoop for saving parquets
        dm_folder - folder for parquets in hdfs_path
        dm_filter_cond - filter for clients you take from dm using SQL-condition syntax (ex.: feature_1 > 1 and feature_2 = 10 and ...)
        transformer_path - local path to pickled transformer, which trainsform categorical features in pandas dataframe by calling transformer.transform, and returns dataframe with same columns
        batch_size - buckets length for dm (default value is suitable in my cases (150-240mb per bucket))

        """
        self.spark = spark
        self.sc = self.spark.sparkContext
        self.hdfs_path = hdfs_path
        self.dm_folder = dm_folder
        self.dm_name = dm_name
        self.numeric_id_name = numeric_id_name
        self.print_func = print_func
        self.batch_size = batch_size
        self.ch_parq_table_func = ch_parq_table_func
        self.transformer_path = transformer_path
        self.dm_filter_cond = dm_filter_cond
        self.dm_path = dm_path
        self.features_to_transform = features_to_transform
        self.max_batch_size = max_batch_size_for_encoding
        self.own_module = own_module
        self.hive_query_func = hive_query_func
        self.dm_periods_column_name = dm_periods_column_name
        self.periods_from_dm = periods_from_dm
        self.slice_date = slice_date
        self.slice_date_pattern = slice_date_pattern
        self.features_to_transform = features_to_transform
        self.score_on_ready_data_mart = score_on_ready_data_mart
        self.f_cols = features_from_dm
        self.cols_to_avg = features_for_avg_aggr

        self.sc.setLogLevel('FATAL')

        self.ex_tab_length, self.dm_length, self.df_enc, self.repart_val = (False,) * 4

        if not self.score_on_ready_data_mart:
            self.tab_for_predict = f'{self.hdfs_path}/{self.dm_folder}/{self.dm_name}'
        else:
            self.tab_for_predict = self.dm_path

        self.tab_for_predict_enc = self.tab_for_predict + '_encoded'
        self.transformer = joblib.load(self.transformer_path)

        if not self.features_to_transform:
            try:
                self.features_to_transform = self.transformer.cat_features
            except AttributeError:
                return Exception('you should set "features_to_transform"')

        if create_tables_in_hive:
            self.hive_database_for_dms = hive_database_for_dms
        else:
            self.hive_database_for_dms = False

        self.sc.addFile(self.transformer_path)

        self.transform_func = '''@F.pandas_udf('bigint')
def transform_{col_name}(*df_col):
    if not os.path.exists('{own_module}'):
        import zipfile
        with zipfile.ZipFile(f'{own_module}.zip', 'r') as z:
            z.extractall('{own_module}')
    transformer = joblib.load('{transformer_path}'.split('/')[-1])
    df_col = pd.DataFrame(df_col[0])
    df_col.columns = ['{col_name}']
    l = len(df_col)
    df_col = transformer.transform(df_col)
    return df_col.iloc[:, 0]'''

    def add_months(self,
                   some_date: str,
                   months_diff: int) -> str:
        from dateutil.relativedelta import relativedelta
        out = dt.datetime.strptime(some_date, self.slice_date_pattern)
        out += relativedelta(months=months_diff)
        return dt.datetime.strftime(out, self.slice_date_pattern)

    def get_date_partition(self, months_range: int) -> str:
        out = [self.add_months(self.slice_date, -x) for x in range(0, months_range)]
        return "'" + "','".join(out) + "'"

    def build_mdm_slice(self):
        if self.score_on_ready_data_mart:
            if not self.ch_parq_table_func(self.tab_for_predict):
                out = f'ERROR: specified dm {self.tab_for_predict} does not exist'
                self.print_func(out)
                raise Exception(out)
        elif not self.ch_parq_table_func(self.tab_for_predict):

            self.print_func(f'building dm {self.tab_for_predict}')

            if self.dm_filter_cond:
                self.dm_filter_cond += ' and '
            else:
                self.dm_filter_cond = ''

            df_dm = self.spark.read.parquet(f'{self.dm_path}').filter(
                f'{self.dm_periods_column_name} in ({self.get_date_partition(max(self.periods_from_dm))})'
            ).select([self.numeric_id_name, *list(set(self.cols_to_avg) | set(self.f_cols))])

            df_1 = df_dm.filter(
                f'{self.dm_filter_cond} {self.dm_periods_column_name} = \'{self.slice_date}\' '
            )

            self.dm_length = df_1.count()
            self.print_func(f'slice length: {self.dm_length}')

            if len(self.cols_to_avg) > 0:
                arrg_features = [
                    df_dm.filter(
                        f'{self.dm_filter_cond} {self.dm_periods_column_name} in ' \
                        f'({self.get_date_partition(period)})'
                    ).select([self.numeric_id_name, *self.cols_to_avg]) \
                        .groupby(self.numeric_id_name) \
                        .agg(*[
                        F.avg(col).alias(f'{col}_{period}a')
                        for col in self.cols_to_avg
                    ])
                    for period in self.periods_from_dm
                ]
            else:
                arrg_features = []
            self.repart_val = int(self.dm_length / self.batch_size)

            repart_dfs = [
                x.repartition(self.repart_val, F.col(self.numeric_id_name)) for x in
                (df_1, *arrg_features)
            ]

            df = repart_dfs.pop(0)

            for another_df in repart_dfs:
                df = df.join(another_df, on=self.numeric_id_name)

            df.repartition(self.repart_val) \
                .write.mode('overwrite') \
                .option('compression', 'none') \
                .parquet(self.tab_for_predict)

            self.spark.catalog.clearCache()
            sh.hdfs('dfs', '-chmod', '-R', '777', self.tab_for_predict)
            sh.hdfs('dfs', '-setrep', '-R', '2', self.tab_for_predict)

        else:
            self.print_func(f'dm {self.dm_name} already exists')

        if self.hive_database_for_dms:
            self.hive_query_func(
                query=f"create database if not exists {self.hive_database_for_dms}; "
                      f"drop table if exists {self.hive_database_for_dms}.{self.dm_name}; "
                      f"create external table {self.hive_database_for_dms}.{self.dm_name} " \
                      f"({','.join([f'{x[0]} {x[1]}' for x in self.spark.read.parquet(self.tab_for_predict).dtypes])}) " \
                      f"row format serde 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' " \
                      f"stored as inputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' " \
                      f"outputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat' " \
                      f"location '{self.tab_for_predict}' "
            )
            self.print_func(f'table {self.hive_database_for_dms}.{self.dm_name} created in hive')

    def transform_spark_df(self,
                           sdf: pyspark.sql.dataframe.DataFrame,
                           path_to_write: str,
                           parquet_write_mode: str = 'append',
                           repartition_val: int = 22) -> None:
        for categ_col in [x for x in sdf.columns if x in self.features_to_transform]:

            if f'transform_{categ_col}' not in globals().keys() or True:
                tr_func = self.transform_func.format(
                    transformer_path=self.transformer_path,
                    col_name=categ_col,
                    own_module=self.own_module
                )

                exec(f'global transform_{categ_col}\n{tr_func}')

            sdf = eval(f'''sdf.withColumn(
                '{categ_col}',
                transform_{categ_col}(*[F.lower(F.regexp_replace(F.col('{categ_col}'), ' ', ''))]))'''
                       )

        sdf = sdf.cache()

        sdf.repartition(repartition_val).write \
            .mode(parquet_write_mode) \
            .option('compression', 'none') \
            .parquet(path_to_write)

    def go_transform(self, parquet_write_mode: str = 'append'):

        if self.ex_tab_length > 0:
            df_enc = self.spark.read.parquet(self.tab_for_predict_enc)
        else:
            df_enc = False

        df_to_enc = self.spark.read.parquet(self.tab_for_predict)

        if not self.dm_length:
            self.dm_length = df_to_enc.count()
            self.repart_val = int(self.dm_length / self.batch_size)

        if self.ex_tab_length == 0:
            cur_lenght = self.dm_length
        else:
            df_to_enc = self.df_to_enc.repartition(self.repart_val.F.col(self.numeric_id_name)).join(
                df_enc.repartition(self.repart_val.F.col(self.numeric_id_name)),
                on=[self.numeric_id_name],
                how='left_anti'
            )
            cur_lenght = df_to_enc.count()

        n_parts = int(np.ceil(cur_lenght / self.max_batch_size))
        if n_parts > 1:
            parts = [np.round(1 / n_parts, 2), ] * n_parts + [np.round(1 / n_parts, 2)]
            df_parts = df_to_enc.randomSplit(parts)
        else:
            df_parts = [df_to_enc, ]

        self.print_func(f'dm n_parts {len(df_parts)}')
        self.print_func('trainsforming...')

        for df_part in df_parts:
            self.transform_spark_df(
                sdf=df_part,
                path_to_write=self.tab_for_predict_enc,
                parquet_write_mode=parquet_write_mode,
                repartition_val=int(cur_lenght / self.batch_size / n_parts),
            )
            self.spark.catalog.clearCache()
            parquet_write_mode = 'append'  # if firts was 'overwrite'

        sh.hdfs('dfs', '-chmod', '-R', '777', self.tab_for_predict_enc)
        sh.hdfs('dfs', '-setrep', '-R', '2', self.tab_for_predict_enc)

        if self.hive_database_for_dms:
            self.hive_query_func(
                query=f"drop table if exists {self.hive_database_for_dms}.{self.tab_for_predict_enc.split('/')[-1]}; "
                      f"create external table {self.hive_database_for_dms}.{self.tab_for_predict_enc.split('/')[-1]} " \
                      f"({','.join([f'{x[0]} {x[1]}' for x in self.spark.read.parquet(self.tab_for_predict_enc).dtypes])}) " \
                      f"row format serde 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' " \
                      f"stored as inputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' " \
                      f"outputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat' " \
                      f"location '{self.tab_for_predict_enc}' "
            )

    def build_dataset(self):
        global tr_is_done

        self.build_mdm_slice()

        df = self.spark.read.parquet(self.tab_for_predict)

        if not self.dm_length:
            self.dm_length = df.count()
        self.print_func(f'dm_length: {self.dm_length}')

        self.print_func(f'tab_for_predict: {self.tab_for_predict}')

        self.repart_val = int(self.dm_length / int(73e4))
        df = df.repartition(self.repart_val)

        ex_of_tr_table = self.ch_parq_table_func(self.tab_for_predict_enc)

        if ex_of_tr_table:
            df_enc = self.spark.read.parquet(self.tab_for_predict_enc)
            self.ex_tab_length = df_enc[[self.numeric_id_name]].count()
            self.print_func(f'ex_tab_length: {self.ex_tab_length}')
        else:
            self.ex_tab_length = 0

        tr_eff = 1

        while self.ex_tab_length < self.dm_length and tr_eff < 20:
            self.print_func(f'transforming {self.tab_for_predict} started, try: {tr_eff}')
            try:
                if not ex_of_tr_table:
                    self.go_transform(parquet_write_mode='overwrite')
                elif self.ex_tab_length < self.dm_length:
                    self.go_transform(parquet_write_mode='append')
                else:
                    self.print_func('transformed dm already exists')
                ex_of_tr_table = self.ch_parq_table_func(self.tab_for_predict_enc)
                self.ex_tab_length = self.spark.read.parquet(self.tab_for_predict_enc) \
                    .select(self.numeric_id_name).count()
                self.print_func(f'ex_tab_length: {self.ex_tab_length}')
            except Exception as E:
                tr_eff += 1
                ex_of_tr_table = self.ch_parq_table_func(self.tab_for_predict_enc)
                max_batch_size_old = self.max_batch_size
                self.max_batch_size = int(self.max_batch_size * 0.5)
                self.print_func(
                    'problems with transforming ' \
                    f'max_batch_size decreased from {max_batch_size_old} ' \
                    f'to {self.max_batch_size} '\
                    'ERROR: ' + str(E).replace("\n", " | ")
                )
                if self.max_batch_size <= int(5e6):
                    self.print_func(f'ERROR: problems with transforming {self.ex_tab_length}, break')
                    sys.exit()


        del df
        gc.collect()
        self.spark.catalog.clearCache()


        if self.ex_tab_length >= self.dm_length:
            tr_is_done = True

        self.sc.stop()
