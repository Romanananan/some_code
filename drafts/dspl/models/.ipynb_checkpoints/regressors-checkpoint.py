"""
# models/regressors.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

=============================================================================

Модуль с реализацией моделей машинного обучения для DS-Template для задачи
регрессии.

Доступные сущности:
- fair_obj: функция потерь fair-loss для XGBRegressor.
- XGBRegressorModel: модель XGBoostRegressor для DS-Template.
- CBRegressorModel: модель CatBoostRegressor для DS-Template.
- LGBRegressorModel: модель LightGBMRegressor для DS-Template.

=============================================================================

"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
from .base import BaseRegressor
from .eval_metrics import rmsle, mse, r2


fair_constant = 1000
def fair_obj(preds, dtrain):
    labels = dtrain.get_label()
    x = (preds - labels)
    den = abs(x) + fair_constant
    grad = fair_constant * x / (den)
    hess = fair_constant * fair_constant / (den * den)
    return grad, hess


# ключ - название метрики
# значение - реализация метрики и флаг максимизации метрики
eval_metrics = {
    "mse": (mse, False),
    "rmsle": (rmsle, False),
    "r2": (r2, True)
}


class XGBRegressorModel(BaseRegressor):
    """
    Модель XGBoost со стандартизованным API для DS-Template.

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
    self.estimator: xgboost.core.Booster
        Обученный estimator xgboost.

    self.evals_result: dict
        Словарь с значениями метрик и функции потерь на
        каддой итерации обучения.

    """

    @staticmethod
    def create_dmatrix(data: pd.DataFrame, target: pd.Series) -> xgb.DMatrix:
        """
        Создание xgb.DMatrix для оптимального обучения XGBoost.
        """
        return xgb.DMatrix(data, label=target)

    def _create_eval_set(self,
                         data: pd.DataFrame,
                         target: pd.Series,
                         *eval_set):
        """
        Создание eval_set в xgb-формате.
        """
        data = self.validate_input_data(data)
        dtrain = self.create_dmatrix(data, target)

        if eval_set:
            valid = self.validate_input_data(eval_set[0])
            dvalid = self.create_dmatrix(valid, eval_set[1])
            return [(dtrain, "train"), (dvalid, "valid")]

        return [(dtrain, "train")]

    def check_metrics(self):
        """
        Проверка eval_metric и objective.
        Если заданная eval_metric реализована в xgboost, то
        она используется, иначе - используется собственная
        реализация.

        """
        objective = self.params.get("objective", "reg:squarederror")
        self.metric_name = self.params.get("eval_metric")
        self.params_ = deepcopy(self.params)
        try:
            feval, maximize = eval_metrics.get(self.metric_name)
            _ = self.params_.pop("eval_metric")
        except TypeError:
            feval, maximize = None, False

        if objective.lower() == "mae":
            _ = self.params_.pop("objective")
            return feval, maximize, fair_obj
        else:
            self.params_["objective"] = "reg:squarederror"
            return feval, maximize, None

    def check_loss(self):
        """
        Проверка objective.
        """
        objective = self.params.get("objective", "reg:squarederror")
        if objective.lower() == "mae":
            _ = self.params.pop("objective")
            return fair_obj
        else:
            self.params["objective"] = "reg:squarederror"
            return None

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

        eval_set: Tuple[pd.DataFrame, pd.Series]
            Кортеж с валидационными данными. Первый элемент
            кортежа - матрица признаков, второй элемент
            кортежа - вектор целевой переменной.

        """
        data = self.validate_input_data(data)
        dtrain = self.create_dmatrix(data, target)
        devals = self._create_eval_set(data, target, *eval_set)

        print(
            (f"{time.ctime()}, start fitting XGBoost-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.\n")
        )

        self.evals_result = {}
        feval, maximize, objective = self.check_metrics()

        self.estimator = xgb.train(
            self.params_, dtrain, evals=devals,
            obj=objective, feval=feval, maximize=maximize,
            num_boost_round=1000, early_stopping_rounds=50,
            verbose_eval=10, evals_result=self.evals_result
        )

    def transform(self, data: pd.DataFrame) -> None:
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
        data = self.create_dmatrix(data, None)

        if isinstance(self.delta_scores, dict):
            n_iters = self.n_iterations
        else:
            n_iters = 0

        prediction = self.estimator.predict(data, ntree_limit=n_iters)
        return prediction

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
        scores = pd.DataFrame()

        for sample_name in self.evals_result:
            scores[sample_name] = self.evals_result[sample_name][self.metric_name]

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = np.abs(scores[train_score] - scores[valid_score])
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

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


class CBRegressorModel(BaseRegressor):
    """
    Модель CatBoostRegressor со стандартизованным API для DS-Template.

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

    def check_loss(self):
        """
        Проверка objective.
        """
        objective = self.params.get("loss_function", "MAE")
        objective = objective.upper()
        if objective == "MSE":
            self.params["loss_function"] = "RMSE"
        else:
            self.params["loss_function"] = "MAE"

    def check_eval_metric(self):
        """
        Проверка eval_metric.
        """
        eval_metric = self.params.get("eval_metric", "MAE")
        self.params["eval_metric"] = eval_metric.upper()
        if eval_metric == "mse":
            self.params["eval_metric"] = "RMSE"

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
        self.check_loss(), self.check_eval_metric()
        data = self.validate_input_data(data)
        print(
            (f"{time.ctime()}, start fitting CatBoost-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.\n")
        )
        fit_params = self._create_fit_params(data, target, *eval_set)
        self.estimator = cb.CatBoostRegressor(**self.params)
        self.estimator.fit(data, target, self.categorical_features, **fit_params)
        self.fitted = True

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

        prediction = self.estimator.predict(data, ntree_end=n_iters)
        return prediction

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
        eval_metric = self.params["eval_metric"]
        scores = pd.DataFrame()

        for sample_name in eval_scores:
            if sample_name != "learn":
                scores[sample_name] = eval_scores[sample_name][eval_metric]

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = np.abs(scores[train_score] - scores[valid_score])
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

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


class LGBRegressorModel(BaseRegressor):
    """
    Модель LightGBMRegressor со стандартизованным API для DS-Template.

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
            "eval_metric": "mae",
            "early_stopping_rounds": 100,
            "eval_set": self._create_eval_set(data, target, *eval_set),
            "verbose": 10
        }

    def check_loss(self):
        """
        Проверка objective.
        """
        objective = self.params.get("objective", "mae")
        objective = objective.lower()
        if objective == "mae":
            self.params["objective"] = "regression_l1"
        else:
            self.params["objective"] = "regression"

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
            (f"{time.ctime()}, start fitting LightGBM-Model, "),
            (f"train.shape: {data.shape[0]} rows, {data.shape[1]} cols.\n")
        )
        self.check_loss()
        fit_params = self._create_fit_params(data, target, *eval_set)
        self.estimator = lgb.LGBMRegressor(**self.params)
        self.estimator.fit(data, target, self.categorical_features, **fit_params)
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

        prediction = self.estimator.predict(data, num_iteration=n_iters)
        return prediction

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
            scores[sample_name] = eval_scores[sample_name]["l1"]

        if scores.shape[1] == 2:
            train_score, valid_score = scores.columns
            scores["absolute_diff"] = np.abs(scores[train_score] - scores[valid_score])
            scores["relative_diff"] = 100 * scores["absolute_diff"] / scores[train_score]

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

