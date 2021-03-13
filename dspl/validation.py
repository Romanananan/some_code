# coding=utf-8
"""
# validation.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

=============================================================================

Модуль с реализацией трансформера для разбиения выборки на train / valid / test.

Доступные сущности:
- DataSplitter: сплиттер на train / valid / [test].

=============================================================================

"""

from typing import List, Dict, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split


class DataSplitter(BaseEstimator, TransformerMixin):
    """
    Разбиение выборки на train / valid / [test].

    Parameters
    ----------
    split_fractions: List[float|int]
        Список с параметрами разбиения выборки на train / valid / [test].
        Может быть указано в виде долей разбиения: [0.6, 0.3, 0.1] или
        в виде количества наблюдений: [200000, 100000, 50000].

    split_column: string, optional, default = None
        Название поля, по которому произвести разбиение.
        Опциональный параметр, по умолчанию не используется.

    """
    def __init__(self, split_fractions: List, split_column: str = None):
        self.split_fractions = split_fractions
        self.n_samples = len(split_fractions)
        self.split_column = split_column

    def transform(self, data: pd.DataFrame, target: pd.Series) -> dict:
        """
        Разбиение исходной выборки на train / valid / [test].

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Обучающая выборка.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        eval_set: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, где ключ - название выборки (train / valid / [test]),
            значение - кортеж из пар data, target.

        """
        splitter = self.get_splitter(data, target)
        return splitter(data, target)

    def get_splitter(self, data: pd.DataFrame, target: pd.Series) -> callable:
        """
        Выбор метода разбиения исходной выборки на train / valid / [test].

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Обучающая выборка.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        splitter: callable
            Метод разбиения данных.

        """
        if target.nunique() == 2:
            return self._random_stratify_split
        if self.split_column:
            return self._column_split
        else:
            return self._random_split

    def _random_stratify_split(self, data: pd.DataFrame, target: pd.Series):
        """
        Случайное, стратифицированное по целевой переменной,
        разбиение данных на train / valid / [test]. Применяется в случае, 
        если target - бинарный вектор.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Обучающая выборка.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        eval_set: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, где ключ - название выборки, значение - кортеж
            с матрицей признаков и вектором целевой переменной.

        """
        train_idx, *valid_idx = self._calculate_split_idx(data.index, target)
        return self._to_evalset(data, target, train_idx, *valid_idx)

    def _random_split(self, data: pd.DataFrame, target: pd.Series):
        """
        Случайное разбиение данных на train / valid / [test].
        Применяется в случае, если target - вектор непрерывной целевой
        переменной.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Обучающая выборка.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        eval_set: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, где ключ - название выборки, значение - кортеж
            с матрицей признаков и вектором целевой переменной.

        """
        train_idx, *valid_idx = self._calculate_split_idx(data.index)
        return self._to_evalset(data, target, train_idx, *valid_idx)

    def _column_split(self, data: pd.DataFrame, target: pd.Series):
        """
        Разбиение данных на train / valid / [test] по заданному полю.
        Применяется в случае, когда задано self.split_column и требуется
        разбить данные по полю (например: по клиенту).

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Обучающая выборка.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        eval_set: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, где ключ - название выборки, значение - кортеж
            с матрицей признаков и вектором целевой переменной.

        """
        values = data[self.split_column].unique()
        train_idx, *valid_idx = self._calculate_split_idx(values)

        train_mask = data[self.split_column].isin(train_idx)
        train_idx = data.loc[train_mask].index

        valid_mask = [data[self.split_column].isin(idx) for idx in valid_idx]
        valid_idx = [data.loc[mask].index for mask in valid_mask]

        return self._to_evalset(data, target, train_idx, *valid_idx)

    def _calculate_split_idx(self, idx_array, target=None):
        """
        Вычисление индексов для train / valid / [test] частей.

        Parameters
        ----------
        idx_array: numpy.array
            Индексы для разбиения.

        target: pd.Series, optional, default = None
            Вектор целевой переменной, опциональный параметр,
            по умолчанию не используется.

        Returns
        -------
        idx: Tuple[np.array]
            Кортеж индексов для train / valid / [test] частей.

        """
        train_idx, valid_idx = train_test_split(
            idx_array, train_size=self.split_fractions[0], stratify=target
        )

        if self.n_samples == 3:
            if isinstance(self.split_fractions[0], float):
                size = int(idx_array.shape[0] * self.split_fractions[1])
            else:
                size = self.split_fractions[1]

            if isinstance(target, pd.Series):
                target = target.loc[valid_idx]

            valid_idx, test_idx = train_test_split(
                valid_idx, train_size=size, stratify=target, random_state=10
            )
            return train_idx, valid_idx, test_idx

        return train_idx, valid_idx

    @staticmethod
    def _to_evalset(data, target, train_idx, *valid_idx):
        """
        Создание словаря eval_set, где ключ - название выборки,
        значение - кортеж с матрицей признаков и вектором целевой
        переменной.
        """
        eval_set = {
            "train": (data.loc[train_idx], target.loc[train_idx]),
            "valid": (data.loc[valid_idx[0]], target.loc[valid_idx[0]])
        }

        if len(valid_idx) == 2:
            eval_set["test"] = (
                data.loc[valid_idx[1]], target.loc[valid_idx[1]]
            )
        return eval_set
