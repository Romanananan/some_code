"""
# db.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

=============================================================================

Модуль с реализацией сущностей для взаимодействия с БД.

Доступные сущности:
- TeraDataDB: подключение к БД TeraData.

=============================================================================

"""

from tqdm import tqdm
import pandas as pd
import teradatasql


class TeraDataDB:
    """
    Взаимодействие с Базой Данных (БД) TeraData.

    Предоставляет методы для выполнения SQL-запросов / загрузки данных
    из таблицы в RAM-память / создания новых таблиц в БД.

    Parameters
    ----------
    username: string
        Логин пользователя.

    password: string
        Пароль.

    host: string, optional, default = "tdsb15.cgs.sbrf.ru"
        Хост.

    """
    def __init__(self, username: str, password: str, host="tdsb15.cgs.sbrf.ru"):
        self.username = username
        self.password = password
        self.host = host

    def get_session(self):
        """Создание пользовательской сессии"""
        return teradatasql.connect(
            host=self.host, user=self.username, password=self.password)

    def sql2df(self, query: str, chunksize=100000) -> pd.DataFrame:
        """
        Функция батчевой выгрузки данных из TeraData в pandas.DataFrame.

        Parameters
        ----------
        query: string
            SQL-запрос в виде python-строки.

        chunksize: int, optional, default = 100000
            Размер батча данных.

        Returns
        -------
        data: pandas.DataFrame
            Набор данных.

        """
        db = pd.read_sql(query, self.get_session(), chunksize=chunksize)
        data = pd.DataFrame()
        for batch_data in tqdm(db, leave=False):
            data = pd.concat([data, batch_data])
        return data
    
    def generator(self, query: str, chunksize=100000):
        db = pd.read_sql(query, self.get_session(), chunksize=chunksize)
        for batch_data in tqdm(db, leave=False):
            yield batch_data
