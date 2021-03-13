"""
# models/classifier.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

=============================================================================

Модуль с реализацией моделей машинного обучения для DS-Template для задачи
бинарной классификации.

Доступные сущности:
- XGBClassifierModel: модель XGBoostClassifier для DS-Template.
- CBClassifierModel: модель CatBoostClassifier для DS-Template.
- LGBClassifierModel: модель LightGBMClassifier для DS-Template.
- RFClassifierModel: модель RandomForestClassifier для DS-Template.
- WBAutoML: модель WhiteBox AutoML для DS-Template.

=============================================================================

"""

import time
from typing import List, Optional
from collections import OrderedDict

#import autowoe
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from .base import BaseClassifier


class XGBClassifierModel(BaseClassifier):
    """
    Модель XGBoost со стандартизованным API для DS-Template.

    Parameters
    ----------
    model_params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных переменных.
        Опциональный параметр, по умолчанию, не используется,
        все переменные рассматриваются как числовые переменные.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества между
        метрикой на обучающей выборке и на валидационной выборке.
        Словарь, где ключ - опция ("absolute" / "relative"),
        значение - допустимая разница.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    self.estimator: xgboost.sklearn.XGBClassifier
        Обученный estimator xgboost.

    """

    def _create_eval_set(self, data: pd.DataFrame, target: pd.Series, *eval_set):
        """
        Создание eval_set в xgb-формате.
        """
        data = self.validate_input_data(data)
        if eval_set:
            valid_data = self.validate_input_data(eval_set[0])
            return [(data, target), (valid_data, eval_set[1])]

        return [(data, target)]

    def _create_fit_params(self, data: pd.DataFrame, target: pd.Series, *eval_set):
        """
        Создание параметров обучения в xgb-формате.

        """
        return {
            "eval_metric": "auc",
            "early_stopping_rounds": 100,
            "eval_set": self._create_eval_set(data, target, *eval_set),
            "verbose": 10,
        }

    def fit(self, data: pd.DataFrame, target: pd.Series, *eval_set) -> None:
        """
        Обучение модели на данных data, target.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        eval_set: Tuple[pd.DataFrame, pd.Series]
            Кортеж с валидационными данными. Первый элемент
            кортежа - матрица признаков, второй элемент
            кортежа - вектор целевой переменной.

        """
        data = self.validate_input_data(data)
        print(
            (f"{time.ctime()}, start fitting XGBoost-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.")
        )
        fit_params = self._create_fit_params(data, target, *eval_set)
        self.estimator = xgb.XGBClassifier(**self.params)
        self.estimator.fit(data, target, **fit_params)

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Применение обученной модели к данным data.
        Для применения модели должен быть ранее вызван метод fit
        и создан self.estimator. Если метод fit не был вызван, то
        будет возбуждено исключение .

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (выборка для применения модели).

        Returns
        -------
        prediction: numpy.array, shape = [n_samples, ]
            Вектор с прогнозами модели на данных data.

        """
        self.check_is_fitted
        data = self.validate_input_data(data)
        if isinstance(self.delta_scores, dict):
            n_iters = self.n_iterations
        else:
            n_iters = None

        prediction = self.estimator.predict_proba(data, ntree_limit=n_iters)
        return prediction[:, 1]

    @property
    def scores(self):
        """
        Расчет метрики качества на выборке для обучения и
        выборке для валидации, если она была указана. Расчет
        разницы метрики качества для значением на обучающей
        выборке и валидационной выборке.

        Если задан self.delta_scores, то количество итераций
        алгоритма выбирается исходя из заданных параметров.
        Если возможно ограничить число итераций таким образом,
        чтобы разница метрик на обучении и валидации была меньше
        заданного порога - то устанавливаются ограничения.
        Иначе - выбирается 50 итераций алгоритма.

        """
        self.check_is_fitted
        eval_scores = self.estimator.evals_result()
        scores = pd.DataFrame()

        for sample_name in eval_scores:
            scores[sample_name] = eval_scores[sample_name]["auc"]
            scores[sample_name] = 100 * (2*scores[sample_name] - 1)

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = scores[train_score] - scores[valid_score]
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

            if "absolute_diff" in self.delta_scores:
                max_delta = self.delta_scores["absolute_diff"]
                mask = scores["absolute_diff"] <= max_delta
                scores = scores[mask]

            if "relative_diff" in self.delta_scores:
                max_delta = self.delta_scores["relative_diff"]
                mask = scores["relative_diff"] <= max_delta
                scores = scores[mask]

        return scores

    @property
    def n_iterations(self):
        """
        Вычисление оптимального количества итераций.
        Оптимальное количество итераций подбирается по алгоритму:
        если задан self.delta_scores - то число
        итераций подбирается при преодолении заданных порогов;
        если не задан self.delta_scores и не задан self.tolerance - то
        используются все итерации, которые были построены.

        """
        return len(self.scores)


class CBClassifierModel(BaseClassifier):
    """
    Модель CatBoostClassifier со стандартизованным API для DS-Template.

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
    self.estimator: catboost.core.CatBoostRegressor
        Обученный estimator catboost.

    """
    def _create_eval_set(self,
                         data: pd.DataFrame,
                         target: pd.Series,
                         *eval_set):
        """
        Создание eval_set в catboost-формате.
        """
        data = self.validate_input_data(data)
        if eval_set:
            valid_data = self.validate_input_data(eval_set[0])
            return [(data, target), (valid_data, eval_set[1])]

        return [(data, target)]

    def _create_fit_params(self,
                           data: pd.DataFrame,
                           target: pd.Series,
                           *eval_set):
        """
        Создание параметров обучения в catboost-формате.
        """
        return {
            "eval_set": self._create_eval_set(data, target, *eval_set),
        }

    def fit(self, 
            data: pd.DataFrame,
            target: pd.Series,
            *eval_set) -> None:
        """
        Обучение модели на данных data, target.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        """
        data = self.validate_input_data(data)
        print(
            (f"{time.ctime()}, start fitting CatBoost-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.\n")
        )
        fit_params = self._create_fit_params(data, target, *eval_set)
        self.estimator = cb.CatBoostClassifier(**self.params)
        self.estimator.fit(data, target, self.categorical_features, **fit_params)

    def transform(self, data):
        """
        Применение обученной модели к данным data.
        Для применения модели должен быть ранее вызван метод fit
        и создан self.estimator. Если метод fit не был вызван, то
        будет возбуждено исключение .

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (выборка для применения модели).

        Returns
        -------
        prediction: numpy.array, shape = [n_samples, ]
            Вектор с прогнозами модели на данных data.

        """
        self.check_is_fitted
        data = self.validate_input_data(data)
        if isinstance(self.delta_scores, dict):
            n_iters = self.n_iterations
            if not isinstance(n_iters, int):
                n_iters = 0
        else:
            n_iters = 0

        try:
            prediction = self.estimator.predict_proba(data, ntree_end=n_iters)
        except cb.CatBoostError:
            prediction = self.estimator.predict_proba(data)

        return prediction[:, 1]

    @property
    def scores(self):
        """
        Расчет метрики качества на выборке для обучения и
        выборке для валидации, если она была указана. Расчет
        разницы метрики качества для значением на обучающей
        выборке и валидационной выборке.

        Если задан self.delta_scores, то количество итераций
        алгоритма выбирается исходя из заданных параметров.
        Если возможно ограничить число итераций таким образом,
        чтобы разница метрик на обучении и валидации была меньше
        заданного порога - то устанавливаются ограничения.
        Иначе - выбирается 50 итераций алгоритма.

        """
        self.check_is_fitted
        eval_scores = self.estimator.evals_result_
        scores = pd.DataFrame()

        for sample_name in eval_scores:
            if sample_name != "learn":
                scores[sample_name] = eval_scores[sample_name]["AUC"]
                scores[sample_name] = 100 * (2*scores[sample_name] - 1)

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = scores[train_score] - scores[valid_score]
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

            if "absolute_diff" in self.delta_scores:
                max_delta = self.delta_scores["absolute_diff"]
                mask = scores["absolute_diff"] <= max_delta
                scores = scores[mask]

            if "relative_diff" in self.delta_scores:
                max_delta = self.delta_scores["relative_diff"]
                mask = scores["relative_diff"] <= max_delta
                scores = scores[mask]

        return scores

    @property
    def n_iterations(self):
        """
        Вычисление оптимального количества итераций.
        Оптимальное количество итераций подбирается по алгоритму:
        если задан self.delta_scores - то число
        итераций подбирается при преодолении заданных порогов;
        если не задан self.delta_scores и не задан self.tolerance - то
        используются все итерации, которые были построены.

        """
        return self.scores.index.max()


class LGBMClassifierModel(BaseClassifier):
    """
    Модель LightGBM со стандартизованным API для DS-Template.

    Parameters
    ----------
    model_params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных переменных.
        Опциональный параметр, по умолчанию, не используется,
        все переменные рассматриваются как числовые переменные.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества между
        метрикой на обучающей выборке и на валидационной выборке.
        Словарь, где ключ - опция ("absolute" / "relative"),
        значение - допустимая разница.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    self.estimator: lightgbm.sklearn.LGBMClassifier
        Обученный estimator lightgbm.

    """
    def _create_eval_set(self,
                         data: pd.DataFrame,
                         target: pd.Series,
                         *eval_set):
        """
        Создание eval_set в lgb-формате.
        """
        data = self.validate_input_data(data)
        if eval_set:
            valid_data = self.validate_input_data(eval_set[0])
            return [(data, target), (valid_data, eval_set[1])]
        return [(data, target)]

    def _create_fit_params(self,
                           data: pd.DataFrame,
                           target: pd.Series,
                           *eval_set):
        """
        Создание параметров обучения в lgb-формате.

        """
        return {
            "eval_metric": "auc",
            "early_stopping_rounds": 100,
            "eval_set": self._create_eval_set(data, target, *eval_set),
            "verbose": 10
        }

    def fit(self, data: pd.DataFrame, target: pd.Series, *eval_set) -> None:
        """
        Обучение модели на данных data, target.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        """
        if not self.categorical_features:
            self.categorical_features = "auto"
        data = self.validate_input_data(data)
        print(
            (f"{time.ctime()}, start fitting LightGBM-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.")
        )
        fit_params = self._create_fit_params(data, target, *eval_set)
        self.estimator = lgb.LGBMClassifier(**self.params)
        self.estimator.fit(
            data, target, categorical_feature=self.categorical_features, **fit_params)
        self.fitted = True

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Применение обученной модели к данным data.
        Для применения модели должен быть ранее вызван метод fit
        и создан self.estimator. Если метод fit не был вызван, то
        будет возбуждено исключение .

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (выборка для применения модели).

        Returns
        -------
        prediction: numpy.array, shape = [n_samples, ]
            Вектор с прогнозами модели на данных data.

        """
        self.check_is_fitted
        data = self.validate_input_data(data)
        if isinstance(self.delta_scores, dict):
            n_iters = self.n_iterations
        else:
            n_iters = None

        prediction = self.estimator.predict_proba(data, num_iteration=n_iters)
        return prediction[:, 1]

    @property
    def scores(self):
        """
        Расчет метрики качества на выборке для обучения и
        выборке для валидации, если она была указана. Расчет
        разницы метрики качества для значением на обучающей
        выборке и валидационной выборке.

        Если задан self.delta_scores, то количество итераций
        алгоритма выбирается исходя из заданных параметров.
        Если возможно ограничить число итераций таким образом,
        чтобы разница метрик на обучении и валидации была меньше
        заданного порога - то устанавливаются ограничения.
        Иначе - выбирается 50 итераций алгоритма.

        """
        self.check_is_fitted
        eval_scores = self.estimator.evals_result_
        scores = pd.DataFrame()

        for sample_name in eval_scores:
            scores[sample_name] = eval_scores[sample_name]["auc"]
            scores[sample_name] = 100 * (2*scores[sample_name] - 1)

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = scores[train_score] - scores[valid_score]
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

        if "absolute_diff" in self.delta_scores:
            max_delta = self.delta_scores["absolute_diff"]
            mask = scores["absolute_diff"] <= max_delta
            scores = scores[mask]

        if "relative_diff" in self.delta_scores:
            max_delta = self.delta_scores["relative_diff"]
            mask = scores["relative_diff"] <= max_delta
            scores = scores[mask]

        return scores

    @property
    def n_iterations(self):
        """
        Вычисление оптимального количества итераций.
        Оптимальное количество итераций подбирается по алгоритму:
        если задан self.delta_scores - то число
        итераций подбирается при преодолении заданных порогов;
        если не задан self.delta_scores и не задан self.tolerance - то
        используются все итерации, которые были построены.

        """
        return len(self.scores)


class RFClassifierModel(BaseClassifier):
    """
    Модель RandomForest со стандартизованным API для DS-Template.

    Parameters
    ----------
    model_params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных переменных.
        Опциональный параметр, по умолчанию, не используется,
        все переменные рассматриваются как числовые переменные.

    delta_scores: Dict[string, float], optional, default = None
        Допустимая разница в значении метрики качества между
        метрикой на обучающей выборке и на валидационной выборке.
        Словарь, где ключ - опция ("absolute" / "relative"),
        значение - допустимая разница.
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    self.estimator: sklearn.ensemble.RandomForestClassifier
        Обученный estimator RandomForestClassifier.

    """

    def fit(self, data: pd.DataFrame, target: pd.Series, *eval_set) -> None:
        """
        Обучение модели на данных data, target.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        eval_set: Tuple[pd.DataFrame, pd.Series]
            Кортеж с валидационными данными. Первый элемент
            кортежа - матрица признаков, второй элемент
            кортежа - вектор целевой переменной.

        """
        data = self.validate_input_data(data)
        print(
            (f"{time.ctime()}, start fitting RandomForest-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.")
        )
        self.estimator = RandomForestClassifier(**self.params)
        self.estimator.fit(data.fillna(-9999), target)

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Применение обученной модели к данным data.
        Для применения модели должен быть ранее вызван метод fit
        и создан self.estimator. Если метод fit не был вызван, то
        будет возбуждено исключение .

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (выборка для применения модели).

        Returns
        -------
        prediction: numpy.array, shape = [n_samples, ]
            Вектор с прогнозами модели на данных data.

        """
        self.check_is_fitted
        data = self.validate_input_data(data)
        prediction = self.estimator.predict_proba(data.fillna(-9999))
        return prediction[:, 1]

    @property
    def scores(self):
        pass

    @property
    def n_iterations(self):
        return rf_model.estimator.n_estimators


class WBAutoML(BaseClassifier):
    """
    Модель WhiteBox AutoML со стандартизованным API для DS-Template.

    Parameters
    ----------
    model_params: dict
        Словарь гиперпараметров модели.

    used_features: List[string]
        Список используемых признаков.

    categorical_features: List[string], optional, default = None
        Список категориальных переменных.
        Опциональный параметр, по умолчанию, не используется,
        все переменные рассматриваются как числовые переменные.

    tree_params: collections.OrderedDict
        Сетка параметров для оптимизации LightGBM.
        LightGBM используется для поиска оптимального WOE-биннинга.

    """

    def __init__(self,
                 model_params: dict,
                 used_features: List[str],
                 categorical_features: Optional[List[str]],
                 tree_params) -> None:

        super().__init__(model_params, used_features, categorical_features)
        self.params["tree_dict_opt"] = tree_params

    def _create_fit_params(self, data: pd.DataFrame, target: pd.Series):
        """
        Создание параметров обучения в autowoe-формате.

        """
        max_bin = self.params.pop("max_bin_count")
        features_type = self.prepare_dtypes(data, target.name)
        monotone = {key: "0" for key in features_type.keys()}

        params = {
            "features_type": features_type,
            "max_bin_count": {x: max_bin for x in features_type.keys()},
            "features_monotone_constraints": monotone,
            "target_name": target.name,
            "select_type": None,
            "folds_codding": False}
        
        return params

    def prepare_dtypes(self, data: pd.DataFrame, target_name: str) -> dict:
        """
        Подготовка типов данных для передачи в модель.

        Parameters
        ----------
        data: pd.DataFrame, shape = [n_samples, n_features]
            Набор данных для обучения модели.

        target_name: str
            Название целевой переменной.

        Returns
        -------
        features_types: Dict[str: str]
            Словарь, ключ - название признака,
            значение - тип признака.

        """
        if not self.categorical_features:
            self.categorical_features = {}

        num_features = set(data.columns) - set(self.categorical_features)
        num_features = num_features - set([target_name])

        cat_features = {x: "cat" for x in self.categorical_features}
        num_features = {x: "real" for x in num_features}

        return dict(**num_features, **cat_features)

    def fit(self, data: pd.DataFrame, target: pd.Series) -> None:
        """
        Обучение модели на данных data, target.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (обучающая выборка).

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        """
        fit_params = self._create_fit_params(data, target)
        print(
            (f"{time.ctime()}, start fitting WhiteBox AutoML, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.")
        )
        dtrain = pd.concat([data, target], axis=1)
        self.estimator = autowoe.AutoWoE(**self.params)
        self.estimator.fit(train=dtrain, **fit_params)
        self.used_features = self.estimator.features_fit[0]
        self.fitted = True

    def transform(self, data: pd.DataFrame) -> np.array:
        """
        Применение обученной модели к данным data.
        Для применения модели должен быть ранее вызван метод fit
        и создан self.estimator. Если метод fit не был вызван, то
        будет возбуждено исключение .

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков (выборка для применения модели).

        Returns
        -------
        prediction: numpy.array, shape = [n_samples, ]
            Вектор с прогнозами модели на данных data.

        """
        self.check_is_fitted
        prediction = self.estimator.predict_proba(data)

        return prediction
