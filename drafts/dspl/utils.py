"""
# utils.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

========================================================================

Модуль с утилит, необходимых для работы DS-Template.

========================================================================

Доступные классы:

- MissedColumnError: исключение, которое выбрасывается, если
  в наборе данных не хватает ожидаемых столбцов.
- INFOSaver: сущность для сохранения выходных файлов.

========================================================================

Доступные функции:

- calculate_time_execute: декоратор для замера времени работы функции.
- cleanup_input: удаление неиспользуемых признаков из набора данных.
- get_input: загрузка данных.
- save_data: версионированное сохранение данных.

========================================================================

"""

import os
import sys
import time
import pickle
import json

from copy import deepcopy
from functools import wraps
from typing import Tuple

try:
    from pyspark import SparkContext, SparkConf, HiveContext   

    spark_home = '/opt/cloudera/parcels/SPARK2/lib/spark2'
    os.environ['SPARK_HOME'] = spark_home
    os.environ['PYSPARK_DRIVER_PYTHON'] = 'python'
    os.environ['LD_LIBRARY_PATH'] = '/opt/python/virtualenv/jupyter/lib'

    sys.path.insert(0, os.path.join (spark_home,'python'))
    sys.path.insert(0, os.path.join (spark_home,'python/lib/py4j-0.10.7-src.zip'))

except ModuleNotFoundError:
    pass

import pandas as pd
from sklearn.model_selection import train_test_split
from .db import TeraDataDB


class MissedColumnError(IndexError):
    """
    Класс для идентификации ошибки несоответствия
    ожидаемых и полученных столбцов в pandas.DataFrame
    """
    pass


class INFOSaver:
    """
    Saver для сохранения выходных файлов пайплайна.
    Перед тем, как сохранить данные, объект проверяет наличие  каталога
    path. Если каталог отсутствуют - то создается структура каталогов
    ({path} - {experiment_number} - config / docs / images / models),
    если каталог присутствует, то создается структура
    ({experiment_number} - config / docs / images / models).

    Предназначение каталогов
    ------------------------
    - {path} - каталог, для сохранения выходной информации всех запусков.
    - {path}/{experiment_number} - каталог, для сохранения выходной
                                   информации данного эксперимента.
    - {path}/{experiment_number}/config - каталог для записи конфига.
    - {path}/{experiment_number}/docs - каталог для записи отчета.
    - {path}/{experiment_number}/images - каталог для записи графиков.
    - {path}/{experiment_number}/models - каталог для записи моделей.

    Parameters
    ----------
    models: dict
        Словарь, где ключ - название модели,
        значение - экземпляр модели со стандартизованным
        API для DS-Template.

    config: dict
        Словарь с конфигурацией эксперимента.

    path: string, optional, default = "runs"
        Путь для сохранения выходных файлов эксперимента.

    Attributes
    ----------
    dir_: string
        Путь с номером эксперимента.

    """
    def __init__(self, models: dict, config: dict, path: str = "runs") -> None:
        self.dir_ = None
        self.path = path
        self.config = config
        self.models = deepcopy(models)

        imp = [col for col in self.models if "importance" in col]
        for col in imp:
            _ = self.models.pop(col)

    def create_dir(self) -> None:
        """
        Создание основного каталога для сохранения элемнетов, полученных
        в ходе эксперимента, и подкаталогов для хранения выходных данных
        о конкретном эксперименте. При вызове - осуществляется попытка
        создать каталога self.path и self.path/1/, если каталог уже
        существует - то осуществляется поиск максимального вложенного
        каталога в self.path и создается каталог с номером на 1 больше.

        """
        if self.dir_ is None:
            try:
                os.mkdir(f"{self.path}/")
                os.mkdir(f"{self.path}/1/")
                self.dir_ = f"{self.path}/1"
            except FileExistsError:
                dirs = [
                    int(num) for num in os.listdir(f"{self.path}/")
                    if not str(num).startswith(".")
                ]

                try:
                    self.dir_ = f"{self.path}/{max(dirs) + 1}"
                except ValueError:
                    self.dir_ = f"{self.path}/1"
                os.mkdir(self.dir_)
        for folder in ['models', 'config', 'images', 'docs']:
            try:
                os.mkdir(f"{self.dir_}/{folder}/")
            except FileExistsError:
                pass

    def save(self) -> None:
        """
        Сохранение моделей и трансформера для категориальных признаков.

        """
        self.create_dir()
        for model in self.models:
            estimator = self.models[model]
            pickle.dump(estimator, open(f"{self.dir_}/models/{model}.pkl", "wb"))

        with open(f"{self.dir_}/config/config.json", "w") as file:
            json.dump(self.config, file)


def calculate_time_execute(func):
    """
    Декоратор для замера времени выполнения функции func.

    Типичный пример использования:
    @calculate_time_execute
    def some_function(<arguments>):
        <setup>
        return res

    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        exec_time = round(time.time() - start_time, 3)
        print(f"Func: {func.__name__}, Execution time: {exec_time} sec.")
        return res
    return wrapper


def create_hive_session(conf=None):
    """
    Создание HiveContext.
    Если не заданы параметры спарковской сессии, то
    поднимается дефолтная спарковская сессия с предопределнными
    параметрами.

    Parameters
    ----------
    conf: pyspark.conf.SparkConf
        Объект SparkConf для установки свойств spark.

    Returns
    -------
    sqlc: pyspark.sql.context.HiveContext
        Сессия для выгрузки данных из Hive.

    """
    if not conf:
        conf = SparkConf().setAppName('SparkContextExample')\
            .setMaster("yarn-client")\
            .set('spark.dynamicAllocation.enabled', 'false')\
            .set('spark.driver.maxResultSize','32g')\
            .set('spark.executor.memory','16g')

    sc = SparkContext.getOrCreate(conf=conf)
    sqlc = HiveContext(sc)

    return sqlc


def prepare_dtypes(df) -> pd.DataFrame:
    """
    Приведение типов данных при переводе из
    pyspark.DataFrame в pandas.DataFrame.

    Parameters
    ----------
    df: pyspark.DataFrame
        Набор данных в pyspark.

    Returns
    -------
    df_transformed: pandas.DataFrame
        Набор данных в pandas.

    """
    obj_features = df.dtypes[df.dtypes == "object"].index
    for feature in obj_features:
        try:
            df[feature] = df[feature].astype(float)
        except ValueError as e:
            print(f"{feature}: {e}")
        except TypeError as e:
            print(f"{feature}: {e}")
    return df


def get_hadoop_input(path: str, conf=None) -> pd.DataFrame:
    """
    Выгрузить данные из Hadoop.

    Parameters
    ----------
    path: string
        Путь до данных / SQL-запрос для выгрузки данных.

    Returns
    -------
    data: pandas.DataFrame
        Набор данных.

    """
    sqlc = create_hive_session(conf)
    data = sqlc.table(path)
    data = data.cache().toPandas()
    data = prepare_dtypes(data)

    return data


def get_teradata_input(path: str, config: dict, generator=False, chunksize=100000) -> pd.DataFrame:
    """
    Выгрузить данные из TeraData.

    Parameters
    ----------
    path: string
        Путь до данных / SQL-запрос для выгрузки данных.

    config: dict
        Конфигурационный файл эксперимента.

    Returns
    -------
    data: pandas.DataFrame
        Набор данных.

    """
    loader = TeraDataDB(config["login"], config["password"])
    if "select" in path.lower():
        if generator:
            return loader.generator(path, chunksize)
        else:
            return loader.sql2df(path, chunksize)
    else:
        path = f"select * from {path}"
        if generator:
            return loader.generator(path, chunksize)
        else:
            return loader.sql2df(path, chunksize)


# @calculate_time_execute
def get_input(data_path: str, config: dict, conf=None, generator=False, chunksize=100000) -> pd.DataFrame:
    """
    Загрузка входных данных.
    Загрузка возможных с диска (.csv / .pkl), из TeraData, из Hadoop.

    Parameters
    ----------
    data_path: string
        Ключ в config, по которому расположен путь до выборки с данными.

    config: dict, optional, default = config_file
        Конфигурационный файл эксперимента.

    conf: pyspark.conf.SparkConf
        
    """
    target_name = config.get("target_name")
    file_name = config.get(data_path, None)

    if file_name is None:
        raise ValueError(f"Incorrect path to {data_path} data.")
    elif ".csv" in file_name:
        data = pd.read_csv(file_name)
    elif ".pkl" in file_name:
        data = pd.read_pickle(file_name)
    else:
        data = get_teradata_input(file_name, config, generator, chunksize)
#         try:
#             data = get_hadoop_input(file_name, conf=conf)
#         except (NameError, FileNotFoundError):
#             data = get_teradata_input(file_name, config, generator, chunksize)
    
    if generator:
        return data
    else:
        save_data(data, name=data_path)
        data = data.reset_index(drop=True)
        return data, data[target_name]


def _cleanup_input(data: pd.DataFrame, config: dict) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Очистка датасета data: выделение целевой переменной в отдельную
    сущность, удаление garbage_features, если они указаны в config.

    Parameters
    ----------
    data: pandas.DataFrame, shape = [n_samples, n_features]
        Набор данных, где n_samples объектов и n_features признаков.

    config: dict
        Словарь конфигурационного файла.

    Returns
    -------
    data: pandas.DataFrame
        Матрица признаков для обучения модели.

    target: pandas.Series
        Вектор целевой переменной

    """
    target_name = config.get("target_name")
    drop_columns = config.get("drop_features")

    if target_name:
        target = data[target_name]
        data = data.drop(target_name, axis=1)
    else:
        raise KeyError(f"{target_name} not in data.")

    if drop_columns:
        dropped_cols = set(drop_columns) & set(data.columns)
        data = data.drop(dropped_cols, axis=1)

    return data.reset_index(drop=True), target.reset_index(drop=True)


def cleanup_input(config, **eval_sets):
    """
    Удаление drop_features и target_name из eval_sets.

    Parameters
    ----------
    config: dict
        Конфигурационный файл эксперимента.

    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Словарь с выборками, для которых требуется рассчитать статистику.
        Ключ словаря - название выборки (train / valid / ...), значение -
        кортеж с матрицей признаков (data) и вектором ответов (target).

    """
    drop_features = config.get("drop_features", [])
    drop_features.append(config.get("target_name"))

    for sample in eval_sets:
        data, target = eval_sets[sample]
        eval_sets[sample] = (data.drop(drop_features, axis=1), target)

    return eval_sets


def save_data(data, name: str):
    """
    Сохранение данные и метаинформации данных в папку data.

    Механизм сохранения: осуществляется попытка создания папки
    data, если папка создана - то сохранить data, meta_data.
    Если папка создана не была, то считывать все meta_data и
    сравнивать с текущим набором данных, если data.shape
    совпал хотя бы с одним meta_data, то сохранение данных
    не производить. Если data.shape не совпал не с одним
    meta_data, то сохранить данные.

    Parameters
    ----------
    data: pandas.DataFrame
        Набор данных для сохранения.

    name: string
        Название набора данных.

    """
    name_ = "_".join(name.split("_")[:2])
    print(f"current dataset shape: {data.shape}")
    try:
        # попытка создать папку data
        os.mkdir("data/")
        data.to_pickle(f"data/{name_}_1.pkl")
        pickle.dump(data.shape, open(f"data/{name_}_meta_1.pkl", "wb"))
    except FileExistsError:
        try:
            save_data = []
            names = [name for name in os.listdir("data/") if name_ in name and "meta" in name]
            for n in names:
                meta_data = pickle.load(open(f"data/{n}", "rb"))
                save_data.append(meta_data != data.shape)
                print(f"loaded {n}, shape = {meta_data}")

            if all(save_data):
                data.to_pickle(f"data/{name_}_{len(names)+1}.pkl")
                pickle.dump(data.shape, open(f"data/{name_}_meta_{len(names)+1}.pkl", "wb"))

        except FileNotFoundError:
            # каталог есть, но данных с таким именем нет
            data.to_pickle(f"data/{name_}_1.pkl")
            pickle.dump(data.shape, open(f"data/{name_}_meta_1.pkl", "wb"))


def choose_psi_sample(eval_sets: dict, config: dict) -> dict:
    """
    Выбор выборки для расчета PSI.
    Выбор осуществляется на основании параметра `psi_sample` в
    конфигурационном файле эксперимента. Если значение равно
    `valid` / `test` - то выбирается данная выборка целиком,
    значение равно `OOT` - то выборка разбивается на 2
    непересекающихся выборки, одна из которых используется
    для расчета PSI, другая используется для независимой
    оценки качества.

    Parameters
    ----------
    eval_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]]
        pass

    config: dict
        Словарь с конфигурацией эксперимента.

    Returns
    -------
    eval_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Преобразованный словарь с eval_set.

    psi_sample: pd.DataFrame
        Выборка для расчета PSI.

    """
    psi_sample_name = config.get("psi_sample", "OOT")

    if psi_sample_name in ["valid", "test"]:
        return eval_sets, eval_sets[psi_sample_name][0]

    elif psi_sample_name == "OOT":
        oot_evaluate, oot_psi = train_test_split(
            eval_sets["OOT"][0], train_size=0.5, random_state=1
        )
        oot_target_evaluate, oot_target_psi = train_test_split(
            eval_sets["OOT"][1], train_size=0.5, random_state=1
        )
        eval_sets["OOT"] = (oot_evaluate, oot_target_evaluate)
        eval_sets["OOT_psi"] = (oot_psi, oot_target_psi)

        return eval_sets, oot_psi

    else:
        raise ValueError(
            f"Unknown psi-sample name! Please choose: {eval_sets.keys()}")
