"""
# models/base.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

=============================================================================

Модуль с реализацией интерфейса моделей машинного обучения.

Доступные сущности:
- BaseModel: API ML-модели.
- BaseClassifier: реализовация базового классификатора.
- BaseRegressor: реализация базового регрессора.

=============================================================================

"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from copy import deepcopy

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.exceptions import NotFittedError
from sklearn.metrics import roc_auc_score

from ..utils import MissedColumnError
from ..metrics import root_mean_squared_error
from ..feature_extraction.utils import calculate_permutation_feature_importance


class BaseModel(ABC, BaseEstimator, TransformerMixin):
    """
    API ML-модели для использования в DS-Template.

    Используется как базовый класс для реализации конкретной модели.
    Содержит общие методы, которые используются для любого типа модели.

    Parameters
    ----------
    params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых для обучения признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных признаков.
        Опциональный параметр, по умолчанию не используется.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества на
        обучающей выборке и на валидационной выборке.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    estimator: callable
        Экземпляр обученной модели.

    """
    def __init__(self,
                 params: dict,
                 used_features: List[str],
                 categorical_features: Optional[List[str]] = None,
                 delta_scores: Optional[dict] = None):

        self.params = deepcopy(params)
        self.used_features = used_features
        if categorical_features:
            self.categorical_features = list(
                set(categorical_features) & set(used_features)
            )
        else:
            self.categorical_features = None
        if delta_scores:
            self.delta_scores = delta_scores
        else:
            self.delta_scores = {}
        self.estimator = None

    def validate_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Проверка входных данных data на наличие требуемых признаков.
        Если ожидаемые признаки отсутствуют в наборе данных, то
        возбуждается MissedColumnError.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для проверки.

        Returns
        -------
        data_validated: pandas.DataFrame
            Матрица признаков, содержащая требуемые признаки.

        """
        if self.used_features:
            missed_features = list(
                set(self.used_features) - set(data.columns)
            )
            if missed_features:
                raise MissedColumnError(f"Missed {list(missed_features)} columns.")
            return data[self.used_features]

        return data

    @property
    def check_is_fitted(self):
        """
        Проверка была ли обучена модель.
        Если проверка не пройдена - возбуждается исключение NotFittedError.
        """
        if not bool(self.estimator):
            msg = ("This estimator is not fitted yet. Call 'fit' with"
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    @abstractmethod
    def fit(self, data: pd.DataFrame, target: pd.Series, *eval_set) -> None:
        """
        Абстрактный метод - обучение модели на данных (data, target).
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> None:
        """
        Абстрактный метод - применение модели к данным data.
        """
        pass


class BaseClassifier(BaseModel):
    """
    Базовый классификатор в DS-Template.

    Используется как базовый класс для реализации конкретного
    классификатора. Содежрит общие методы, которые используется
    для любой реализации классификатора и не зависят от деталей
    реализации.

    Parameters
    ----------
    params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых для обучения признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных признаков.
        Опциональный параметр, по умолчанию не используется.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества на
        обучающей выборке и на валидационной выборке.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    estimator: callable
        Экземпляр обученной модели.

    """

    def evaluate_model(self, **eval_sets) -> Tuple[float]:
        """
        Оценка качества модели метрикой GINI на eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь, где ключ - название выборки, значение - кортеж с
            матрицей признаков и вектором истинных ответов.

        """
        self.check_is_fitted
        scores = {}
        for sample in eval_sets:
            data, target = eval_sets[sample]
            prediction = self.transform(data)

            try:
                score = roc_auc_score(target, prediction)
                score = 2*score - 1
                score = 100*score
            except ValueError:
                score = 0
            
            scores[sample] = score
            
            print(f"{sample}-score:\t GINI = {round(score, 2)}")
            
        return scores['train'], scores['test'], scores['test'] - scores['train']
        
    def feature_importance(self, data, target):
        """
        Расчет важности признаков на основе перестановок.
        Важность рассчитывается, если задан self.eval_set,
        и применен метод `fit` для модели. Если self.eval_set
        не задан, то возбуждается ValueError.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        feature_importance: pandas.DataFrame
            Оценка важности признаков.

        """
        self.check_is_fitted
        return calculate_permutation_feature_importance(
            self, roc_auc_score, data[self.used_features], target)


class BaseRegressor(BaseModel):
    """
    Базовый регрессор в DS-Template.

    Используется как базовый класс для реализации конкретного
    регрессора. Содежрит общие методы, которые используется
    для любой реализации регрессора и не зависят от деталей
    реализации.

    Parameters
    ----------
    params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых для обучения признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных признаков.
        Опциональный параметр, по умолчанию не используется.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества на
        обучающей выборке и на валидационной выборке.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    estimator: callable
        Экземпляр обученной модели.

    """

    def evaluate_model(self, **eval_sets) -> None:
        """
        Оценка качества модели метриками MAE, R2, RMSE на eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь, где ключ - название выборки, значение - кортеж с
            матрицей признаков и вектором истинных ответов.

        """
        self.check_is_fitted
        for sample in eval_sets:
            data, target = eval_sets[sample]
            prediction = self.transform(data)

            try:
                rmse = root_mean_squared_error(target, prediction)
                mae = mean_absolute_error(target, prediction)
                r2 = r2_score(target, prediction)
            except ValueError:
                mae, r2, rmse = 0, 0, 0

            res = (f"{sample}-score:\t MAE = {round(mae, 0)}, "
                   f"R2 = {round(r2, 2)}, RMSE = {round(rmse, 0)}")
            print(res)

    def feature_importance(self, data, target):
        """
        Расчет важности признаков на основе перестановок.
        Важность рассчитывается, если задан self.eval_set,
        и применен метод `fit` для модели. Если self.eval_set
        не задан, то возбуждается ValueError.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        feature_importance: pandas.DataFrame
            Оценка важности признаков.

        """
        self.check_is_fitted

        y_pred = self.transform(data[self.used_features])
        baseline_score = mean_absolute_error(target, y_pred)

        importance = calculate_permutation_feature_importance(
            self, mean_absolute_error, data[self.used_features], target, maximize=False)
        importance["permutation_importance"] = baseline_score - importance["permutation_importance"]
        importance["permutation_importance"] = importance["permutation_importance"] / baseline_score
        importance["permutation_importance"] = 1 - importance["permutation_importance"]

        return importance
