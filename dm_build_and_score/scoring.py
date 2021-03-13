import pickle, time, joblib, pyspark, pandas as pd, re, gc, os, numpy as np, sh, sys
from typing import Any
from pyspark.sql import functions as F
from pyspark.sql import types as T


class Scoring:

    def __init__(self,
                 spark: pyspark.sql.session.SparkSession,
                 hdfs_path: str,
                 scores_folder: str,
                 score_date: str,
                 models_path: str,
                 models_npv_path: str,
                 numeric_id_name: str,
                 skip_ex_scores: bool,
                 tab_for_predict_enc: str,
                 tab_for_predict_raw: str,
                 periods: str,
                 print_func: Any,
                 ch_file_ex_func: Any,
                 ch_parq_table_func: Any,
                 needed_bps_npv: list,
                 bucket_size: int,
                 periods_file='periods.xlsx',
                 needed_couples = False
                 ):
        self.spark = spark
        self.sc = self.spark.sparkContext
        self.hdfs_path = hdfs_path
        self.scores_folder = scores_folder
        self.score_date = score_date
        self.models_path = models_path
        self.models_npv_path = models_npv_path
        self.numeric_id_name = numeric_id_name
        self.skip_ex_scores = skip_ex_scores
        self.tab_for_predict_enc = tab_for_predict_enc
        self.periods = periods
        self.print_func = print_func
        self.problem_models, self.scored_models = [], []
        self.df = self.spark.read.parquet(tab_for_predict_enc)
        self.df_raw = self.spark.read.parquet(tab_for_predict_raw)
        self.dm_length = self.df.count()
        self.periods_file = periods_file
        self.used_features = []
        self.needed_bps_npv = needed_bps_npv
        self.bucket_size = bucket_size
        self.ch_parq_table_func = ch_parq_table_func
        self.ch_file_ex_func = ch_file_ex_func
        self.needed_couples = needed_couples

        self.sc.setLogLevel('FATAL')

        if not self.periods:
            self.periods = pd.read_excel(periods_file)
            self.periods['model'] = self.periods.apply(lambda x: f"{x['bp']}_{x['sp']}", axis=1)
            self.periods = self.periods[['model', 'T_min', 'T_max']].set_index('model').T.to_dict()
            self.periods_setted = None
        else:
            self.periods_setted = {'T_min': periods[0], 'T_max': periods[-1]}

        if not needed_couples:
            try:
                config_file = f'{self.hdfs_path}/score_config/cltv_resp_product_pairs_to_score.csv'
                df = self.spark.read.option('delimiter', ';').csv(
                    config_file, header=True
                ).toPandas().fillna(-1).set_index('BP').astype('int')
                self.needed_couples = []
                for col in df.columns[1:]:
                    self.needed_couples.extend([
                        (bp, col, priority) for bp, priority in
                        df[df[col] > 0][['priority']].reset_index().values
                        if priority > 0
                    ])
                self.needed_couples = pd.DataFrame(self.needed_couples, columns=['bp', 'sp', 'p'])\
                    .sort_values(['p', 'bp', 'sp'])[['bp', 'sp']].values.tolist()
                self.needed_couples = list(map(tuple, self.needed_couples))
            except Exception as E:
                self.print_func(f'problem with reading config {config_file}: {E}')
                sys.exit()
        self.needed_bps_npv.sort()

        self.score_f = '''@F.pandas_udf(T.ArrayType(T.DoubleType()))
def uplifts_{bp}_{sp}(*cols):
    if not os.path.exists('dspl'):
        import zipfile
        with zipfile.ZipFile(f'dspl.zip', 'r') as z:
            z.extractall('dspl')
    with open('{model_file_f}', 'rb') as f:
        model = pickle.load(f)
    df = pd.concat(cols, axis = 1).astype('float32')
    del cols
    gc.collect()
    l = df.shape[0]
    df.columns = {used_features}
    if str(model).startswith('CBClassifier'):
        cut_ct_ft = list(set(model.categorical_features)&set(df.columns))
        if len(cut_ct_ft) > 0:
            df[cut_ct_ft] = df[cut_ct_ft].astype('int64')
    output = pd.DataFrame()
    nans = np.empty(l)
    nans[:] = np.nan
    for t in range(0, 13):
        if ('{bp}'=='{sp}' or ('{bp}'=="kp" and '{sp}'=="pl"))  and t==0:
            output[str(t)] = nans
        elif t in range({start_t}, {end_t}):
            df['T'] = t
            upl = []
            for prchs in range(2):
                df['{bp}'] = prchs
                upl.append(np.round(model.transform(df).astype('float32'),5))
            output[str(t)] = upl[1] - upl[0]
        else:
            output[str(t)] = nans
    return pd.Series(output[[str(x) for x in
        sorted([int(x) for x in output.columns.tolist()])
        ]].values.tolist())'''

        self.score_npv_f = '''@F.pandas_udf(T.DoubleType())
def score_npv_{model_name}(*cols):
    if not os.path.exists('dspl'):
        import zipfile
        with zipfile.ZipFile(f'dspl.zip', 'r') as z:
            z.extractall('dspl')
    with open('{model_file_f}', 'rb') as f:
        model = pickle.load(f)
    df = pd.concat(cols, axis = 1)
    del cols
    gc.collect()
    df.columns = {used_features}
    if '{encoder_file}' != 'False':
        with open('{encoder_file}', 'rb') as f:
            encoder = pickle.load(f)
        df = encoder.transform(df)
        gc.collect()
    return pd.Series(model.transform(df.astype('float32'))).round(2)'''

        if not self.ch_file_ex_func(f'{hdfs_path}{self.scores_folder}'):
            sh.hdfs('dfs', '-mkdir', f'{hdfs_path}{self.scores_folder}')
            sh.hdfs('dfs', '-chmod', '-R', '777', f'{hdfs_path}{self.scores_folder}')

        self.dm_length = self.df.count()
        self.print_func(f'dm_length: {self.dm_length}')

    def get_score_func(self, model_file):

        model_file_f = model_file.split("/")[-1]
        bp, sp = model_file_f.replace('.pkl', '').split('_')
        self.sc.addPyFile(model_file)
        model = joblib.load(model_file)
        self.used_features = model.used_features
        self.used_features = [x for x in self.used_features if x not in ('T', bp)]

        new_score_f = self.score_f.format(
            bp=bp,
            sp=sp,
            model_file=model_file,
            used_features=self.used_features,
            model_file_f=model_file_f,
            start_t=self.periods.get(f'{bp}_{sp}', self.periods_setted)['T_min'],
            end_t=self.periods.get(f'{bp}_{sp}', self.periods_setted)['T_max'] + 1,
        )
        exec(f'global uplifts_{bp}_{sp}\n' + new_score_f)

    def get_score_func_npv(self, model_file, model_encoder_file=False):

        model_file_f = model_file.split("/")[-1]
        model_name = model_file_f.replace('.pkl', '')
        self.sc.addPyFile(model_file)
        model = joblib.load(model_file)
        self.used_features = model.used_features

        if model_encoder_file:
            self.sc.addPyFile(model_encoder_file)

        new_score_f = self.score_npv_f.format(
            model_name=model_name,
            model_file_f=model_file_f.split('/')[-1],
            encoder_file=str(model_encoder_file).split('/')[-1],
            used_features=self.used_features
        )
        exec(f'global score_npv_{model_name}\n' + new_score_f)

    def score_npv(self, bp):
        score_tab_name = f'{self.hdfs_path}/{self.scores_folder}/cltv_get_{self.score_date}_npv_{bp}'

        if self.skip_ex_scores & self.ch_parq_table_func(score_tab_name):
            self.print_func(f'score {bp} already exists: {score_tab_name}')
            return

        if bp != 'dc':
            encoder_path = os.path.join(self.models_npv_path, bp, f'encoder_{bp}.pkl')
            encoder_path = encoder_path if os.path.exists(encoder_path) else False
            self.get_score_func_npv(
                model_file=os.path.join(self.models_npv_path, bp, f'{bp}.pkl'),
                model_encoder_file=encoder_path
            )

            df_to_save = eval(
                f"self.df_raw.select(self.numeric_id_name, score_npv_{bp}(*self.used_features).alias('npv_{bp}'))"
            )
        else:
            encoder_path_dc1 = os.path.join(self.models_npv_path, bp, 'dc1', 'encoder_dc1.pkl')
            encoder_path_dc2 = os.path.join(self.models_npv_path, bp, 'dc2', 'encoder_dc2.pkl')
            encoder_path_dc1 = encoder_path_dc1 if os.path.exists(encoder_path_dc1) else False
            encoder_path_dc2 = encoder_path_dc1 if os.path.exists(encoder_path_dc1) else False
            self.get_score_func_npv(
                model_file=os.path.join(self.models_npv_path, bp, 'dc1', 'dc1.pkl'), model_encoder_file=encoder_path_dc1
            )
            used_features_dc1 = self.used_features
            self.get_score_func_npv(
                model_file=os.path.join(self.models_npv_path, bp, 'dc2', 'dc2.pkl'), model_encoder_file=encoder_path_dc2
            )
            used_features_dc2 = self.used_features
            df_to_save = eval(
                f'''self.df.withColumn('nmb_group', F.lit(1)).select(
                    "{self.numeric_id_name}",
                    F.when(F.col("prd_crd_dc_active_qty") > 0, score_npv_dc2(*used_features_dc2))\
                    .otherwise(score_npv_dc1(*used_features_dc1))\
                    .alias('npv_{bp}')
                )'''
            )
            df_to_save = df_to_save.repartition(int(np.ceil(self.dm_length / self.bucket_size / 3)))
            df_to_save.write.option('compression', 'none').mode('overwrite').parquet(score_tab_name)
            self.print_func(f'score {bp} recorded in {score_tab_name}')
            sh.hdfs('dfs', '-setrep', '-R', '2', score_tab_name)
            sh.hdfs('dfs', '-chmod', '-R', '777', score_tab_name)

    def score_couple(self, bp, sp):

        if (self.skip_ex_scores) & (f'{bp}_{sp}' in self.scored_models):
            return

        self.print_func(f'Starting to score {bp}_{sp}')
        hadoop_path = f'{self.hdfs_path}{self.scores_folder}'
        score_tab_name = f'{hadoop_path}/uplifts_{bp}_{sp}_{self.score_date}'
        aggr_scores_name = f'{hadoop_path}/uplifts_{bp}_{self.score_date}'
        aggr_scores_step_names = [
            f'{hadoop_path}/step_uplifts_{bp}_{self.score_date}_{i}' for i in range(6)
        ]

        if (self.skip_ex_scores) & (self.ch_parq_table_func(score_tab_name)):
            self.print_func(f'score {bp}_{sp} already exists: {score_tab_name} or added in {aggr_scores_name}')
            self.spark.catalog.clearCache()
            return
        else:
            if self.ch_parq_table_func(aggr_scores_name):
                self.scored_models.extend([
                    x for x in self.spark.read.parquet(aggr_scores_name).columns[1:-1]
                    if x not in self.scored_models
                ])
                if f'{bp}_{sp}' in self.scored_models:
                    self.print_func(
                        f'scores {",".join([x for x in self.scored_models if x.startswith(bp+"_")])} '\
                        f'already added in {aggr_scores_name}'
                    )
                    self.spark.catalog.clearCache()
                    return
            rdy_couples = []
            for t in aggr_scores_step_names:
                if self.ch_parq_table_func(t):
                    rdy_couples.extend([
                        x for x in self.spark.read.parquet(t).columns[1:-1]
                        if x not in self.scored_models
                    ])
                    self.scored_models.extend(rdy_couples)
                else:
                    break
                if len(rdy_couples) != 0:
                    if f'{bp}_{sp}' in rdy_couples:
                        self.print_func(f'score {bp}_{sp} already added in {t}')
                        self.spark.catalog.clearCache()
                        return

        if bp != 'dc1':
            model_file = f'{self.models_path}/{bp}_{sp}.pkl'  # SL
            model_file_f = f'{bp}_{sp}.pkl'
            score_f = self.get_score_func(model_file)
            new_col_name = f'{bp}_{sp}'
            df_to_save = eval(
                f'''self.df.select(
                    "{self.numeric_id_name}",
                    uplifts_{bp}_{sp}(*[F.col(x) for x in self.used_features]).alias("{bp}_{sp}")
                ).repartition(int(np.ceil(self.dm_length/self.bucket_size)))'''
            )

        else:
            model_file_f = f'{bp}_{sp}.pkl'
            uplifts_dc2p = self.get_score_func(f'{self.models_path}/dc2p_{sp}.pkl')
            used_features_dc2p = [F.col(x) for x in self.used_features]
            uplifts_dc1 = self.get_score_func(f'{self.models_path}/dc1_{sp}.pkl')
            used_features_dc1 = [F.col(x) for x in self.used_features]

            df_to_save = eval(
                f'''self.df.select(
                    "{self.numeric_id_name}",
                    F.when(F.col("prd_crd_dc_active_qty") > 0, uplifts_dc2p_{sp}(*used_features_dc2p))\
                    .otherwise(uplifts_dc1_{sp}(*used_features_dc1))\
                    .alias("{bp}_{sp}")
                ).repartition(int(np.ceil(self.dm_length/self.bucket_size)))'''
            )

        self.print_func(f'scoring {bp}_{sp}')

        df_to_save = df_to_save.repartition(int(np.ceil(self.dm_length / self.bucket_size / 2.5)))

        df_to_save.write.option('compression', 'none').mode('overwrite').parquet(score_tab_name)
        sh.hdfs('dfs', '-setrep', '-R', '2', score_tab_name)
        sh.hdfs('dfs', '-chmod', '-R', '777', score_tab_name)
        self.print_func(f'scores {bp}_{sp} recorded to {score_tab_name}')
        gc.collect()
        self.spark.catalog.clearCache()
        self.scored_models.append(f'{bp}_{sp}')
        self.print_func(
            f'{round((self.needed_couples.index((bp, sp)) + 1) / len(self.needed_couples) * 100, 2)}' \
            f'% of response models done'
        )

    def score_couples(self):

        repart_val = int(self.dm_length / int(73e4))
        self.df = self.df.repartition(repart_val)

        attempts_limit, score_attempts, score_of_couple = 5, 0, False

        self.print_func(f'couples for npv-score: {self.needed_bps_npv}')

        for npv_bp in self.needed_bps_npv:
            while not score_of_couple and score_attempts < attempts_limit:
                try:
                    self.score_npv(npv_bp)
                    score_of_couple, score_attempts = True, 0
                    self.print_func(
                        f'{round((self.needed_bps_npv.index(npv_bp) + 1) / len(self.needed_bps_npv), 2) * 100}'\
                        '% of npv models done'
                    )
                except Exception as E:
                    e = str(E).replace('\n', ' | ')
                    score_attempts += 1
                    if not score_of_couple:
                        self.print_func(f'problem with model {npv_bp} : {e}')
                    if score_attempts >= attempts_limit:
                        self.problem_models.append(f'{npv_bp}')
                        score_of_couple, score_attempts = True, 0
                        continue
                    time.sleep(300)

            score_of_couple = False

        score_attempts = 0

        self.print_func(f'couples for score: {self.needed_couples}')
        for bp, sp in self.needed_couples:
            while not score_of_couple and score_attempts < attempts_limit:
                try:
                    self.score_couple(bp, sp)
                    score_of_couple, score_attempts = True, 0
                except Exception as E:
                    e = str(E).replace('\n', ' | ')
                    score_attempts += 1
                    if not score_of_couple:
                        self.print_func(f'problem with model {bp}_{sp} : {e}')
                    if score_attempts >= attempts_limit:
                        self.problem_models.append(f'{bp}_{sp}')
                        score_of_couple, score_attempts = True, 0
                        continue
                    time.sleep(300)
            score_of_couple = False