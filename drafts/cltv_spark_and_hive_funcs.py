import sh, re, os, sys, socket
sys.path.insert(0, os.path.abspath('../../../utils'))
spark_home = '/opt/cloudera/parcels/SPARK2/lib/spark2'
os.environ['SPARK_HOME'] = spark_home
os.environ['PYSPARK_DRIVER_PYTHON'] = '/opt/venvs/anaconda/bin/python'
os.environ['PYSPARK_PYTHON'] = '/opt/venvs/anaconda/bin/python'
os.environ['LD_LIBRARY_PATH'] = '/opt/python/virtualenv/jupyter/lib'
sys.path.insert(0, os.path.join (spark_home,'python'))
sys.path.insert(0, os.path.join (spark_home,'python/lib/py4j-0.10.7-src.zip'))
import pyspark
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql import Window


jdbc_driver_and_link = \
    'jdbc:hive2://pklis-ldrb00058.labiac.df.sbrf.ru:10000/default;principal=hive/_HOST@DF.SBRF.RU'
ps_path = 'hdfs://clsklsbx/user/team/team_ds_cltv/'

def check_hive_table_existence(tab_name: str):
    spark.sql(f"use {tab_name.split('.')[0]}")
    if spark.sql(f"show tables like '{tab_name.split('.')[1]}'").count() > 0:
        return True
    else:
        return False

def get_table_path(tab_name: str):
    return re.findall(
        r'hdfs://\S+',
        send_beeline_query(f'desc formatted {tab_name}', False, True)
    )[0]


def drop_hive_table(tab_name, and_folder = True, in_ps = False):
    global spark
    if check_hive_table_existence(tab_name):
        if all((in_ps, and_folder)):
            cr_tb_data = spark.sql(
                f'show create table {tab_name}'
            ).collect()[0][0]
            spark.sql(f'show create table {tab_name}')
            tab_location = get_table_path(tab_name)
            sh.hadoop('fs', '-rm', '-skipTrash', '-r', tab_location)
    spark.sql(f'drop table if exists {tab_name}')
    if and_folder and not in_ps:
        os.system(
            'hdfs dfs -rm -r -skipTrash '\
            '/user/hive/warehouse/{}.db/{}'.format(*tab_name.split('.'))
        )


def read_hive_table(tab_name:str):
    spark.catalog.refreshTable(tab_name)
    return spark.read.table(tab_name)        


def send_beeline_query(
    query: str,
    print_output: bool = True,
    return_output: bool = False,
    output_query_in_file: bool = False,
    output_log_in_file: bool = False,
    output_file_names: list = ['tmp_beeline_out.txt', 'tmp_beeline_out.log']):
    
    if output_query_in_file:
        output_file_name = ' 1> ' + output_file_names[0]
    else: output_file_name = ''
    if output_log_in_file:
        output_file_name += ' 2> '+ output_file_names[1]
    else: output_file_name = ''
    if print_output or return_output:
        sh_input = '-u',"'jdbc:hive2://pklis-ldrb00058.labiac.df.sbrf.ru:10000/default;principal=hive/_HOST@DF.SBRF.RU'",\
            '-e','"{}"'.format(query)
        out = sh.beeline(*sh_input)
        if print_output: print(out)
        if return_output: return out.stdout.decode('utf-8')
    else:
        os.system(
            "beeline -u 'jdbc:hive2://pklis-ldrb00058.labiac.df.sbrf.ru:10000/default;principal=hive/_HOST@DF.SBRF.RU' "\
            f'-e "{query}" {output_file_name}'
        )


def save_sdf_to_ps(sdf: pyspark.sql.dataframe.DataFrame or bool = False,
                   table_name: str = 'new_tab',
                   cur_path: str or bool = False,
                   overwrite: bool = True,
                   hive_schema: str = 'default',
                   ps_folder: str = '',
                   parquet_write_mode: str = 'overwrite',
                   parquet_compression: str = 'none',
                   ps_path: str = 'hdfs://clsklsbx/user/team/team_ds_cltv/'):
    """sdf - Spark DataFrame to save
    table_name - new table name in Hive
    overwrite - overwriting Hive table if it exists
    hive_schema - name of Hive db
    ps_folder - directory in "Persistent Storage" to save
    ps_path - hdfs-link to our "Persistent Storage"
    cur_path - if files exist, we only creating external table
    """
    tab_name = f'{hive_schema}.{table_name}'
    existence = check_hive_table_existence(tab_name)
    ps_folder = hive_schema if len(ps_folder) == 0 else ps_folder
    final_path = f'{ps_path}{ps_folder}'
    table_path = f'{final_path}/{table_name}'

    if any([not existence, overwrite]):
        if existence:
            if not cur_path:
                sh.hadoop('fs', '-rm', '-skipTrash', '-r', table_path)
            else:
                sh.hadoop('distcp', cur_path, new_path)
                sh.hadoop('fs', '-rm', '-skipTrash', '-r', table_path)
            drop_hive_table(tab_name, False)
    else:
        print(f'{tab_name} already exists')
        return None

    if cur_path:
        sdf = spark.read.parquet(cur_path)
        table_path = cur_path

    for column in sdf.dtypes:
        if 'date' in column[1]:
            sdf = sdf.withColumn(
                column[0], F.col(column[0]).cast(T.TimestampType()).alias(column[0])
            )
    if not cur_path:

        if len(ps_folder) > 0:
            hadoop_folders = list(filter(
                lambda x: len(x) > 1,
                sh.hadoop('fs', '-ls', '-C', ps_path).split('\n')
            ))
            hadoop_folders = [x.split('/')[-1] for x in hadoop_folders]
            if not any([x == ps_folder for x in hadoop_folders]):
                sh.hadoop('fs', '-mkdir', final_path)
                sh.hdfs('dfs', '-chmod', '-R', '777', final_path)
		
        sdf.write.option('compression', parquet_compression) \
            .mode(parquet_write_mode).parquet(table_path)

    sh.hdfs('dfs', '-setrep', '-R', '2', table_path)

    send_beeline_query(
        query=f"create external table {tab_name} " \
              f"({','.join([f'{x[0]} {x[1]}' for x in sdf.dtypes])}) " \
              f"row format serde 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe' " \
              f"stored as inputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat' " \
              f"outputformat 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat' " \
              f"location '{table_path}' ",
        print_output=False
    )

    sh.hdfs('dfs', '-chmod', '-R', '777', table_path)
    print(f'{tab_name} created, files based in {table_path}')


def unpack_dspl(file_name = 'dspl', strictly = False):
    import zipfile
    if strictly:
        if os.path.exists(file_name):
            os.system(f'rm -r {file_name}')
    if not os.path.exists(file_name) or strictly:
        with zipfile.ZipFile(f'{file_name}.zip', 'r') as z:
            z.extractall(file_name)

def transform_spark_df(sdf, table_name, ps_folder = '', partition_col = False, parquet_write_mode = 'append'):
        
    for categ_col in [x for x in sdf.columns if x in transformer.cat_features]:
        
        @F.pandas_udf('bigint')
        def transform_col(*df_col):
            unpack_dspl()            
            transformer = joblib.load(transformer_path.split('/')[-1])
            df_col = pd.DataFrame(df_col[0])
            df_col.columns = [categ_col]
            l = len(df_col)
            df_col = transformer.transform(df_col)
            return df_col.iloc[:,0]

        sdf = sdf.withColumn(
            categ_col, 
            transform_col(*[F.regexp_replace(F.col(categ_col), ' ', '')])
        )
    
    if partition_col:
        sdf = sdf.drop(partition_col)
    
    save_sdf_to_ps(
        sdf, 
        table_name = table_name.split('.')[1],
        overwrite = False,
        hive_schema = table_name.split('.')[0],
        ps_folder = ps_folder,
        parquet_write_mode = parquet_write_mode
    )
    
    spark.catalog.clearCache()
    
    return table_name

def check_hdfs_file_ex(ps_path: str) -> bool:
    import subprocess
    proc = subprocess.Popen(
        ['hadoop', 'fs', '-test', '-e', ps_path]
    )
    proc.communicate()
    rc = proc.returncode
    if rc != 0: return False
    else: return True

def get_parquets_from_sdf(sdf: pyspark.sql.dataframe.DataFrame):
    name = 'tmp_file' + f'{os.getpid()}_{socket.gethostname().replace(".", "")}'
    while os.path.exists(name):
        name += '_'
    if check_hdfs_file_ex(name):
        sh.hdfs('dfs', '-rm', '-r', '-skipTrash', '{}'.format(name))
    for column in sdf.dtypes:
        if 'date' in column[1]:
            sdf = sdf.withColumn(
                column[0], F.col(column[0]).cast(T.TimestampType()).alias(column[0])
            )
    sdf.write.mode('overwrite').parquet(name)
    sh.hdfs('dfs', '-get', '{}'.format(name), '{}'.format(os.getcwd()))
    sh.hdfs('dfs', '-rm', '-r', '-skipTrash', '{}'.format(name))
    data = pd.read_parquet(name+'/')
    os.system(f'rm -r {os.getcwd()}/{name}')
    return data

def get_repartition_value(
        sdf: pyspark.sql.dataframe.DataFrame,
        target_size: int = 245,
        compression: str = 'none') -> int:
    lenght = sdf.count()
    df_1_row = sdf.limit(int(1e4))
    tmp_file_name = 'test_file'
    while check_hdfs_file_ex(tmp_file_name): tmp_file_name+='_'
    df_1_row.coalesce(1).write.option('compression', compression)\
        .mode('overwrite').parquet(tmp_file_name)
    row_byte_weight = int(sh.hdfs('dfs', '-du', tmp_file_name)\
        .stdout.decode('utf-8').split('\n')[-2].split(' ')[0])
    sh.hdfs('dfs', '-rm', '-R', '-skipTrash', tmp_file_name)
    nd_rep_val = int(row_byte_weight * lenght / target_size / (1024*1024) / 1e4 )
    return 1 if nd_rep_val < 1 else nd_rep_val

def repair_hdfs_table(ps_path, schema, tabname):

    bash_ls = f'hdfs fsck {ps_path}/{schema}/{tabname} -locations -blocks -files | grep "user.*parquet: CORRUPT"'
    p = subprocess.Popen(
        bash_ls, stdout=subprocess.PIPE, shell=True
    )
    partitions, _ = p.communicate()
    partitions = [s.decode('utf-8') for s in partitions.split(b'\n')[:-1]]
    partitions = [s.split(' ')[0].split('/')[-1][:-1] for s in partitions]
    for prt in partitions:
        sh.hdfs('dfs', '-rm', '-skipTrash',  f'{ps_path}/{schema}/{tabname}/{prt}')
    sh.hdfs('hdfs', 'dfs', '-setrep', '2', f'{ps_path}/{schema}/{tabname}')


