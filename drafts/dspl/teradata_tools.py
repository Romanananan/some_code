import os
import numpy as np
import pandas as pd
import random
import platform
import subprocess
import shlex
import teradata
from joblib import dump
import shutil
from tqdm import tqdm


def get_session(db, usr, pwd):
    """Функция устанавливает соединение с ТД и возвращает сессию"""

    if platform.system() == 'Windows':
        driver = 'Teradata'
    else:
        driver = 'Teradata Database ODBC Driver 16.20'

    udaExec = teradata.UdaExec(appName='DataLoad', version='0.1', logConsole=False)
    session = udaExec.connect(method='odbc',
                              system=db,  # Сервер ТД из файла
                              username=usr,  # Логин TD
                              password=pwd,  # Пароль TD
                              driver = driver,
                              charset='UTF8',
                              autoCommit='True',
                              USEREGIONALSETTINGS='N',
                              transactionMode = 'TERADATA'
                              )
    return session


def sql2df(query, session, chunksize=100000):
    """ Функция грузит из терадаты данные в батчах по 100к и склеивает их в одну таблицу """
    db = pd.read_sql(query, session, chunksize=chunksize)
    data = pd.DataFrame()
    for x in tqdm(db):
        data = pd.concat([data, x])
    return data


def check_config():
    """ .twbcfg.ini to root path """
    path = os.path.expanduser("~")
    config_path = os.path.join(path, ".twbcfg.ini")
    log_path = os.path.join(path, "tmp", "teradata_logs")

    if not os.path.exists(config_path):
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        config = f'''CheckpointDirectory='{log_path}' 
    LogDirectory='{log_path}' '''
        with open(config_path, 'w') as f:
            f.write(config)



def td_download(query="",
                bd="tdsb15.cgs.sbrf.ru",
                username="", password="",
                fast=False, return_df=False, csv=True,
                chunksize=100000):
    """
    Функция возвращает данные из ТД: путь к csv или датафрейм.

    fast=True - использовать утилиты ТД, False - ODBC;
    return_df - вернуть датафрейм;
    csv - записать данные в файл при fast=False;
    chunksize - размер бача для ODBC;
    query должен содержать where, чтобы выгрузить название столбцов из БД

    """
    local_seed = str(random.randint(0, 1000000))
    query = query.replace("\n", " ")

    if not fast:
        # Teradata python package
        session = get_session(bd, username, password)
        frame = sql2df(query, session, chunksize=chunksize)
        session.close()
        if return_df:
            return frame
        else:
            path_to_file = os.path.join(os.getcwd(), 'data', 'input_' + local_seed)
            if csv:
                filename = path_to_file + ".csv"
                frame.to_csv(filename, sep=';', index=False, encoding="utf8")
                return filename
            else:
                dump(frame, path_to_file)
                return path_to_file
    else:
        # FastLoad
        check_config()
        query = query.replace("'", "''")  # prepair query for FastLoad
        path_to_folder = os.path.join(os.getcwd(), 'data', 'input_' + local_seed)

        if os.path.exists(path_to_folder):
            shutil.rmtree(path_to_folder)
            os.mkdir(path_to_folder)
        else:
            os.mkdir(path_to_folder)

        path_to_file = os.path.join(path_to_folder, 'dataset.csv')
        open(path_to_file, 'w').close()

        # Create utility files
        txt = '''SourceTdpId = '%s'
               ,SourceUserName = '%s' 
               ,SourceUserPassword = '%s'
               ,DDLPrivateLogName = 'ddlprivate.log'
               ,ExportPrivateLogName = 'exportprivate.log'
               ,TargetErrorList = ['3807']
               ,TargetFileName = '%s'
               ,TargetFormat = 'delimited'
               ,TargetTextDelimiter = ';'
               ,TargetOpenMode = 'write'
               ,SelectStmt = '%s' ''' % (bd, username, password, path_to_file, query)
        qtxt = '''USING CHAR SET UTF-8
               DEFINE JOB qstart2
               (
                 APPLY TO OPERATOR ($FILE_WRITER)
                 SELECT * FROM OPERATOR($EXPORT);
               );'''
        with open(path_to_folder + '/qstart2.txt', 'w+') as f:
            f.write(qtxt)
        with open(path_to_folder + '/jobvars.txt', 'w+') as f:
            f.write(txt)
        # run FastLoad
#         p = subprocess.Popen(
#             shlex.split(f"tbuild -f {path_to_folder}/qstart2.txt -v {path_to_folder}/jobvars.txt -j qstart2")
#         )
#         p.wait()
        p = subprocess.run(
            shlex.split(f"tbuild -f {path_to_folder}/tdd.txt -v {path_to_folder}/jobvars.txt -j tdd_{str(local_seed)}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )

        # columns names
        query = query.replace("\n", " ").replace("''","'")
        query = query.lower()
        query_list = query.split("where")
        if len(query_list) == 2:
            columns_query = " where 1=0 and ".join(query_list)
            session = get_session(bd, username, password)
            columns_names = pd.read_sql(columns_query, session).columns.tolist()
            session.close()
        else:
            print("Coudn't load columns names")
            columns_names = None

        if not return_df:
            if columns_names:
                with open(path_to_folder + '/columns_names.txt', 'w') as f:
                    f.write("\n".join(columns_names))
            return path_to_file
        else:
            if columns_names:
                frame = pd.read_csv(path_to_file, names=columns_names, delimiter=';')
            else:
                frame = pd.read_csv(path_to_file, header=None, delimiter=';')
            return frame


def py2td(x):
    """Функция вставляет пропуски и корректирует тип данных под ТД"""
    x_type = type(x)
    if x_type == float:
        if x % 1 == 0:
            return int(x)
        else:
            return x
    elif x == 'null':
        return None
    else:
        return x


def td_import(
        username="", password="",
        bd="tdsb15.cgs.sbrf.ru", tbl_name="",
        schema="SBX_RETAIL_MP_PFM",
        loadframe=True, df=None, path_to_file=None, fast=False,
        batch_size=12000, max_sessions=6, buffersize=524288,
):
    """
    Функция записывате данные в ТД через утилиты или ODBC

    """
    table = schema + "." + tbl_name
    if not fast:
        if not loadframe:
            df = pd.read_csv(path_to_file, sep=';', encoding='utf8', index=False)
        # insert
        n_iters = len(df) // batch_size + (len(df) % batch_size > 0)
        df_dict = df.to_dict('records')
        session = get_session(bd, username, password)
        for i in tqdm(range(n_iters), total=n_iters):
            session.executemany(
                f"INSERT INTO {table} VALUES ({','.join(list('?' * df.shape[1]))})",
                [list(row.values()) for row in df_dict[i * batch_size:i * batch_size + batch_size]],
                batch=True
            )
        session.close()
    else:
        check_config()
        local_seed = str(random.randint(0, 1000000))
        path_to_folder = os.path.join(os.getcwd(), "data", "output_" + local_seed)

        if os.path.exists(path_to_folder):
            shutil.rmtree(path_to_folder)
        else:
            os.mkdir(path_to_folder)

        if loadframe:
            converted = df.replace(np.NaN, '').astype(str)
            path_to_file = path_to_folder + '/tmp.csv'
            converted.to_csv(path_to_file, index=False, header=False, sep=";", encoding="utf8")
            converted_len = converted.apply(lambda x: x.str.encode('utf-8').apply(len)).max().to_dict()
        else:
            converted_len = pd.read_csv(path_to_file, sep=';', dtype="str", header=None, encoding="utf8",
                                        low_memory=False, nrows=100000)
            columns_query = f"select * from {table} where 1=0"
            session = get_session(bd, username, password)
            columns_names = pd.read_sql(columns_query, session).columns.tolist()
            session.close()
            shutil.copy(path_to_file, path_to_folder + "/tmp.csv")  # cp file for correct working Change to move&

            converted_len.columns = columns_names
            converted_len = converted_len.apply(lambda x: x.str.encode('utf-8').apply(len)).max().to_dict()

        # create empty tmp table
        td_temp_table = table + "_tmp_" + local_seed  # change schema
        session = get_session(bd, username, password)
        session.execute(
            f"create multiset table {td_temp_table} as {table} with no data no primary index"
        )
        session.close()
        # Create utility file
        txt = f"""USING CHARACTER SET UTF8
        DEFINE JOB teradata_upload
        Description 'Fastload script'
        (
            DEFINE OPERATOR Load_operator
            TYPE LOAD
            SCHEMA *
            ATTRIBUTES
            (
                VARCHAR TdPid='{bd}',
                VARCHAR UserName='{username}',
                VARCHAR UserPassWord='{password}',
                VARCHAR TargetTable='{td_temp_table}',
                VARCHAR LogTable='{schema}.usr_tpt_log',
                VARCHAR DateForm='AnsiDate',
                INTEGER MaxSessions={max_sessions}
            );

            DEFINE SCHEMA Define_Employee_Schema
            (
                {','.join(f'{key} VARCHAR({max(1, value*2)})' for key, value in converted_len.items())} 
            );

            DEFINE OPERATOR Producer_File_Detail
            TYPE DATACONNECTOR PRODUCER
            SCHEMA Define_Employee_Schema
            ATTRIBUTES
            (
                VARCHAR DirectoryPath='{path_to_folder}/'
                , VARCHAR FileName='tmp.csv'
                , VARCHAR TextDelimiter=';'
                , VARCHAR QuotedData = 'Optional'
                , VARCHAR OpenQuoteMark = '"'
                , VARCHAR CloseQuoteMark = '"'
                , VARCHAR Format='Delimited'
                , VARCHAR OpenMode='Read'
                , VARCHAR INDICATORMODE='N'
                , INTEGER BUFFERSIZE = {buffersize}
            );

            APPLY
            (
               'INSERT INTO {td_temp_table}({','.join(
            f'{key}' for key, value in converted_len.items())}) VALUES (:{',:'.join(
            f'{key}' for key, value in converted_len.items())});'
            )
            TO OPERATOR(Load_operator)

            SELECT * FROM OPERATOR (Producer_File_Detail);
        );"""
        with open(path_to_folder + '/load_code.tpt', 'w+') as f:
            f.write(txt)
        # Start TPT load
        p = subprocess.Popen(
            shlex.split(f"tbuild -f {path_to_folder}/load_code.tpt -L {path_to_folder}")
        )
        p.wait()
        # Merge
        print("Merging in Teradata...   \r", end='', flush=True)
        session = get_session(bd, username, password)
        session.execute(f"insert into {table} sel * from {td_temp_table}")
        session.close()
        # Drop temporary table
        print("Cleaning...              \r", end='', flush=True)
        session = get_session(bd, username, password)
        session.execute(f"drop table {td_temp_table}")
        session.close()
        # Cleanup
        shutil.rmtree(path_to_folder)
        print("Done!")
