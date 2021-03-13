#!/opt/venvs/anaconda/bin/python

#instead of "print"
def write_to_hdfs_log(msg: str = '' )->None:
    global global_error
    msg = str(msg).replace('\n', ' | ')
    msg = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S') \
          + '\t' + str(msg)
    print(msg)
    with open(local_log_file, 'a') as f:
        f.write('\n' + msg)
    os.system(
        'echo "{}" | hdfs dfs -appendToFile - {}' \
            .format(msg, logs_path)
    )

def del_table(table_path: str, in_hive: bool = False )->None:
    tab_name = '.'.join(table_path.split('/')[-2:])
    if in_hive:
        send_beeline_query(f'drop table if exists {tab_name}')
    os.system('hdfs dfs -rm -r -skipTrash {}'.format(table_path))

def send_beeline_query(query: str,
                       print_output: bool = True,
                       return_output: bool = False )->None or str:
    if print_output or return_output:
        sh_input = '-u', beeline_link, \
                   '-e', '"{}"'.format(query)
        out = sh.beeline(*sh_input).stdout.decode('utf-8')
        if print_output: write_to_hdfs_log(out)
        if return_output: return out
    else:
        os.system(
            f"beeline -u {beeline_link} " \
            f'-e "{query}" '
        )

def check_hdfs_file_ex(file_path: str)-> bool:
    proc = subprocess.Popen(
        ['hdfs', 'dfs', '-test', '-e', file_path]
    )
    proc.communicate()
    rc = proc.returncode
    if rc != 0: return False
    else: return True

def check_hdfs_table_ex(table_path: str) -> bool:
    sf = f'{table_path}/_SUCCESS'
    return check_hdfs_file_ex(sf)

if __name__ == '__main__':

    try:

        global_error = False

        import datetime as dt, joblib, os, pandas as pd, re, sys, sh \
            ,numpy as np, gc, argparse, time, subprocess, warnings

        warnings.filterwarnings('ignore')

        import datetime as dt, joblib, os, pandas as pd, re, sys, sh \
            ,numpy \
            as np, gc, argparse, time, subprocess

        sys.path.append('./dspl')
        sys.path.append('./dm_build_and_score')

        bash_args = argparse.ArgumentParser()
        bash_args.add_argument(
            '--slice_date', type=str, default='False',
            help='date of slice from data mart (will be added to name) (if \'False\' -> last existed date) (default=\'False\')'
        )
        bash_args.add_argument(
            '--id_col_name', type=str, default='epk_id',
            help='name of col with client id (default=\'epk_id\')'
        )
        bash_args.add_argument(
            '--dm_source', type=str, default='hdfs://clsklod/data/core/dwh/sbx_retail_mp_ext/pa/ft_client_aggr_mnth_epk',
            help='data mart name (default=\'hdfs://clsklod/data/core/dwh/sbx_retail_mp_ext/pa/ft_client_aggr_mnth_epk\')'
        )
        bash_args.add_argument(
            '--hadoop_path', type=str, default='hdfs://clsklsbx/user/team/team_ds_cltv/',
            help='link to hadoop folder for saving scores (default=\'hdfs://clsklsbx/user/team/team_ds_cltv/\')'
        )
        bash_args.add_argument(
            '--hive_db_for_dm', type=str, default='avt',
            help='hive db for saving data mart (default=\'avt\')'
        )
        bash_args.add_argument(
            '--dm_folder', type=str, default='dms',
            help='hadoop folder to save data marts (default=\'dms\')'
        )
        bash_args.add_argument(
            '--skip_ex_scores', type=str, default='True',
            help='skip ready scores (default=\'True\')'
        )
        bash_args.add_argument(
            '--score_on_ready_data_mart', type=str, default='False',
            help='(put name of prepared data here) score on ready data, without building slice from major dm (default=\'False\')'
        )
        bash_args.add_argument(
            '--score_exec_cnt', type=int, default=18,
            help='amount of scoring executors'
        )
        bash_args.add_argument(
            '--dataset_already_encoded', type=str, default='False',
            help='is ready dm already encoded?'
        )
        bash_args.add_argument(
            '--periods', type=str, default='False',
            help='T-period for scoring'
        )
        bash_args.add_argument(
            '--logs_path', type=str, default='hdfs://clsklsbx/user/team/team_ds_cltv/logs/scoring_logs/',
            help='folder for logging in Hadoop'
        )
        bash_args.add_argument(
            '--log_file_mask', type=str, default='log',
            help='mask of log file name'
        )
        bash_args.add_argument(
            '--python_kernel', type=str, default='/opt/venvs/anaconda/bin/python',
            help='path of python which we will use'
        )
        bash_args.add_argument(
            '--beeline_node_and_port', type=str, default='pklis-ldrb00058.labiac.df.sbrf.ru:10000',
            help='host and port for beeline (bash hive client)'
        )
        bash_args.add_argument(
            '--needed_base_npv', type=str, default='["dc"]',
            help='base product for scoring personal npv default=["dc"]'
        )
        bash_args.add_argument(
            '--needed_response_couples', type=str, default='False',
            help='needed response couples for scoring, default=False, format: [("ap", "cc"), ("sc", "h"), ...]'
        )
        bash_args.add_argument(
            '--scores_folder', type=str, default='False',
            help='folder for scores on selected hadoop-path, if False -> scores_ps_{slice_date}'
        )
        bash_args.add_argument(
            '--dm_executors_qty', type=int, default=22,
            help='number of executors for building data mart'
        )

        if 'get_ipython' in globals():  # for testing from ipython
            bash_args = bash_args.parse_args("")
        else:
            bash_args = bash_args.parse_args()

        slice_date = bash_args.slice_date
        id_col_name = bash_args.id_col_name
        dm_source = bash_args.dm_source
        ps_path = bash_args.hadoop_path
        dm_folder = bash_args.dm_folder
        skip_ex_scores = eval(bash_args.skip_ex_scores)
        needed_bps_npv = eval(bash_args.needed_base_npv)
        scores_folder = bash_args.scores_folder
        needed_response_couples = eval(bash_args.needed_response_couples)
        dataset_already_encoded = eval(bash_args.dataset_already_encoded)
        periods = eval(bash_args.periods)
        score_on_ready_data_mart = str(bash_args.score_on_ready_data_mart)
        score_exec_cnt = bash_args.score_exec_cnt
        logs_path_catalog = bash_args.logs_path
        log_file_mask = bash_args.log_file_mask
        beeline_node_and_port = bash_args.beeline_node_and_port
        python_kernel = bash_args.python_kernel
        hive_db_for_dm = bash_args.hive_db_for_dm
        dm_executors_qty = bash_args.dm_executors_qty

        beeline_link = f"'jdbc:hive2://{beeline_node_and_port}/" \
                       "default;principal=hive/_HOST@DF.SBRF.RU'"

        local_log_addr = 'log_file_addr.txt'

        if str(score_on_ready_data_mart) == 'False':
            score_on_ready_data_mart = eval(score_on_ready_data_mart)

        today = dt.datetime.today().strftime('%Y%m%d')
        local_log_file = f'temp_local_log.txt'

        logs_path = logs_path_catalog + '' if logs_path_catalog.endswith('/') else '/'
        logs_path += f'{log_file_mask}_{today}.txt'

        with open(local_log_addr, 'w') as f:
            f.write(logs_path)
            f.close()

        write_to_hdfs_log(f'log will be written in {logs_path}')

        if slice_date == 'False' and not score_on_ready_data_mart:
            slice_date = \
                subprocess.check_output(
                    f'hdfs dfs -ls -C {dm_source} | cut -d = -f 2 | grep -P "\\d+-\\d+-\\d+" | tail -1',
                    shell=True
                ).decode('utf-8').strip()

        if not check_hdfs_file_ex(logs_path):
            write_to_hdfs_log(f'scoring is starting {today}')
            sh.hdfs('dfs', '-chmod', '777', logs_path)
            sh.hdfs('dfs', '-setrep', '-R', '2', logs_path)

        if score_on_ready_data_mart:
            dm_folder, final_tab_name = score_on_ready_data_mart.split('/')[-2:]
        else:
            final_tab_name = f'cltv_get_{slice_date.replace("-", "")}_scoring_res'

        models_path = os.path.abspath('./models')
        npv_models_path = os.path.abspath('./models_npv')
        transformer_path = os.path.abspath('./cltv_transformer_full_ld_20210113_3y.pkl')
        periods_file = os.path.abspath('./periods.xlsx')
        numeric_id_name = id_col_name
        db_with_dm = dm_folder
        dm_pattern = '%get_%_scoring_res%'
        if score_on_ready_data_mart:
            ready_data_mart_slice_date = re.findall('[0-9]{8}', score_on_ready_data_mart)

        scores_pref = slice_date.replace("-", "") if not score_on_ready_data_mart else (ready_data_mart_slice_date[-1] if len(ready_data_mart_slice_date)>0 else "")

        if scores_folder == 'False':
            scores_folder = f'scores_ps_{scores_pref}'
        if score_on_ready_data_mart:
            scores_folder = f'scores_ps_{score_on_ready_data_mart.split("/")[-1]}'

        write_to_hdfs_log(f'scores will be written in {ps_path}{scores_folder}')

    except Exception as E:
        write_to_hdfs_log('ERROR: ' + str(E).replace('\n', ' | '))
        write_to_hdfs_log('EXIT')
        sys.exit()

    tries_cnt = 1
    tr_is_done = dataset_already_encoded
    sc_is_done = False

    if os.path.exists(local_log_file):
        os.remove(local_log_file)
    sh.touch(local_log_file)

    if os.path.exists('SUCCESS_'):
        os.remove('SUCCESS_')

    while not all([tr_is_done, sc_is_done]) and tries_cnt < 20:
        try:

            batch_size = int(2.5e5)
            executor_instances = dm_executors_qty
            max_batch_size = int(2e7)
            max_dm_enc_batch = int(8e5)
            max_score_batch_size = int(5e5)
            bucket_size = int(7e6)
            executor_cores = 2
            executor_memory = 10

            sys.path.append('./dspl')
            sys.path.insert(0, os.path.abspath('../../../utils'))
            spark_home = '/opt/cloudera/parcels/SPARK2/lib/spark2'
            os.environ['SPARK_HOME'] = spark_home
            os.environ['PYSPARK_DRIVER_PYTHON'] = python_kernel
            os.environ['PYSPARK_PYTHON'] = python_kernel
            os.environ['LD_LIBRARY_PATH'] = '/opt/python/virtualenv/jupyter/lib'
            sys.path.insert(0, os.path.join(spark_home, 'python'))
            sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.10.7-src.zip'))
            import pyspark
            from pyspark.sql import functions as F
            from pyspark.sql import types as T
            from pyspark import SparkContext, SparkConf

            conf = SparkConf().setAppName('cltv_dm_building_and_transform') \
                .setMaster("yarn-client") \
                .set('spark.yarn.appMasterEnv.ARROW_PRE_0_15_IPC_FORMAT', 1) \
                .set('spark.executorEnv.ARROW_PRE_0_15_IPC_FORMAT', 1) \
                .set('spark.local.dir', 'sparktmp') \
                .set('spark.executor.memory', '{}g'.format(executor_memory)) \
                .set('spark.executor.instances', '{}'.format(executor_instances)) \
                .set('spark.port.maxRetries', '500') \
                .set('spark.executor.cores', '{}'.format(executor_cores)) \
                .set('spark.dynamicAllocation.enabled', 'false') \
                .set('spark.network.timeout', '700') \
                .set("spark.sql.execution.arrow.enabled", "true") \
                .set("spark.kryoserializer.buffer.max", '2047mb') \
                .set("spark.driver.memory", '17g') \
                .set("spark.driver.cores", '12') \
                .set("spark.dynamicAllocation.enabled", "false") \
                .set("hive.exec.dynamic.partition", "true") \
                .set("hive.exec.dynamic.partition.mode", "nonstrict") \
                .set("spark.sql.execution.arrow.maxRecordsPerBatch", '{}'.format(batch_size)) \
                .set("spark.sql.sources.bucketing.maxBuckets", '{}'.format(bucket_size)) \
                .set("have.merge.sparkfiles", "true") \
                .set("spark.files.overwrite", "true") \
                .set("spark.debug.maxToStringFields", "400") \
                .set("spark.python.worker.reuse", "true")

            if not tr_is_done and not dataset_already_encoded:

                dm_name = f'{ps_path}{dm_folder}/{final_tab_name}'
                tab_for_predict = dm_name
                tab_for_predict_enc = tab_for_predict + '_encoded'

                sc = SparkContext.getOrCreate(conf=conf.setAppName('cltv_scoring')) \
                    .addFile(f'{os.getcwd()}/dspl.zip')

                exec(open(os.path.join(spark_home, 'python/pyspark/shell.py')).read())
                sc.setLogLevel('FATAL')
                write_to_hdfs_log('spark session for dm_building opened')

                needed_dm_features_all, features_for_periods = [[
                    x for x in open(f'{file}.txt').read().split(',') if len(x) > 0
                ] for file in ('needed_dm_features_all', 'features_for_periods')]
                
                from dm_build_and_score.dm_building import Dm_building
                
                dm_building = Dm_building(
                    spark=spark,
                    dm_path=dm_source,
                    hdfs_path=ps_path,
                    dm_folder=dm_folder,
                    dm_name=final_tab_name,
                    slice_date=slice_date,
                    numeric_id_name=numeric_id_name,
                    print_func=write_to_hdfs_log,
                    ch_parq_table_func=check_hdfs_table_ex,
                    transformer_path=transformer_path,
                    hive_query_func=send_beeline_query,
                    create_tables_in_hive=False,
                    #hive_database_for_dms='avt',
                    features_from_dm=needed_dm_features_all,
                    features_for_avg_aggr=features_for_periods
                )

                if not tr_is_done:
                    dm_building.build_dataset()
                    tr_is_done = True
                sc.stop()
            else:
                tab_for_predict = dm_source
                tab_for_predict_enc = score_on_ready_data_mart
                write_to_hdfs_log(f'encoding doesn\'t need')
                tr_is_done = True

            write_to_hdfs_log(f'opening score-session')
            executor_memory, executor_cores, executor_instances = \
                10, 3, score_exec_cnt
            conf = conf.set('spark.executor.memory', '{}g'.format(executor_memory)) \
                .set('spark.executor.cores', '{}'.format(executor_cores)) \
                .set('spark.executor.instances', '{}'.format(executor_instances))
            sc = SparkContext.getOrCreate(conf=conf).addFile(f'{os.getcwd()}/dspl.zip')
            exec(open(os.path.join(spark_home, 'python/pyspark/shell.py')).read())
            sc.setLogLevel('FATAL')
            write_to_hdfs_log('spark session for scoring opened')

            from dm_build_and_score.scoring import Scoring
            problem_models = []
            if not sc_is_done:
                try:

                    scoring = Scoring(
                        spark=spark,
                        hdfs_path=ps_path,
                        scores_folder=scores_folder,
                        score_date=scores_pref,
                        models_path=models_path,
                        models_npv_path=npv_models_path,
                        numeric_id_name=numeric_id_name,
                        skip_ex_scores=skip_ex_scores,
                        tab_for_predict_enc=tab_for_predict_enc,
                        tab_for_predict_raw=tab_for_predict,
                        needed_couples=needed_response_couples,
                        periods=periods,
                        print_func=write_to_hdfs_log,
                        ch_file_ex_func=check_hdfs_file_ex,
                        ch_parq_table_func=check_hdfs_table_ex,
                        needed_bps_npv=needed_bps_npv,
                        bucket_size=bucket_size,
                        periods_file='periods.xlsx'
                    )
                    scoring.score_couples()
                    sc_is_done = True
                    problem_models = scoring.problem_models
                    sc.stop()
                except Exception as E:
                    write_to_hdfs_log('ERROR in scoring: ' + str(E).replace('\n', ' | '))

            if len(problem_models) == 0:
                write_to_hdfs_log('BRAVO: scoring completed successfully')
                sh.touch('SUCCESS_')
            else:
                write_to_hdfs_log(
                    f'ENDED WITH PROBLEMS: not scored models: {problem_models}'
                )

            write_to_hdfs_log('EXIT')
            sys.exit()

        except Exception as E:
            global_error = True
            if 'sc' in globals(): sc.stop()
            write_to_hdfs_log('ERROR: ' + str(E).replace('\n', ' | '))
            tries_cnt += 1
            time.sleep(300)

    write_to_hdfs_log('EXIT')
    sys.exit()
