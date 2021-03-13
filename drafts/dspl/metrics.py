"""
# metrics.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

==============================================================================

Модуль с реализацией сущностей для расчета метрик ML-моделей, качества данных.

Доступные сущности:
- calculate_quantile_bins: расчет квантильных бинов.
- gini_score: расчет метрики GINI.

- CalculateDataStatistics: расчет статистики по исследуемым данным.
- CalculateBinaryMetrics: расчет метрик для задачи бинарной классификации.

==============================================================================

"""

import numpy as np
import pandas as pd

from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, confusion_matrix


def calculate_quantile_bins(data: pd.Series, n_bins: int = 20) -> pd.Series:
    """
    Расчет квантильных бакетов.

    Parameters
    ----------
    data: pandas.Series
        вектор значений, для разбиения на квантили.

    n_bins: int, optional, default = 20
        количество бинов, на которые требуется разбить.

    Returns
    -------
    data_transformed: pandas.Series
        квантильные бакеты.

    """
    bins = np.linspace(0, 100, n_bins)
    perc = [np.percentile(data, x) for x in bins]
    perc = np.sort(np.unique(perc))
    return pd.cut(data, perc, labels=False)


def calculate_conf_interval(data, alpha=0.05):
    """
    Вычисление доверительного интервала для среднего.

    Parameters
    ----------
    data: array-like, shape = [n_samples, ]
        Вектор признака для построения интервала.

    alpha: float, optional, default = 0.05
        Уровень доверия.

    Returns
    -------
    conf_interval: Tuple[float, float]
        Границы доверительного интервала.

    """
    y_mean = np.mean(data)
    y_var = np.var(data, ddof=1)

    q_value = stats.t.ppf(1 - alpha/2, data.shape[0])
    std_error = q_value * np.sqrt(y_var) / np.sqrt(data.shape[0])

    return y_mean - std_error, y_mean + std_error


def gini_score(y_true, y_pred) -> float:
    """
    Вычисление метрики GINI.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор значений целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор значений прогнозов.

    Returns
    -------
    score: float
        Значение метрики GINI.

    """
    return 2*roc_auc_score(y_true, y_pred) - 1


def root_mean_squared_error(y_true, y_pred) -> float:
    """
    Вычисление метрики RMSE для задачи регрессии.

    Parameters
    ----------
    y_true: array-like
        Вектор целевой переменной.

    y_pred: array-like
        Вектор прогнозов.

    Returns
    -------
    RMSE: float
        Значение метрики RMSE.

    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_percentage_error(y_true, y_pred) -> float:
    """
    Вычисление метрики MAPE для задачи регрессии.

    Parameters
    ----------
    y_true: array-like
        Вектор целевой переменной.

    y_pred: array-like
        Вектор прогнозов.

    Returns
    -------
    MAPE: float
        Значение метрики MAPE.

    """
    mape = np.abs(y_true - y_pred) / np.abs(1 + y_true)
    valid_mape = mape != np.inf

    mape = np.sum(mape[valid_mape]) / mape[valid_mape].shape[0]
    return mape


def root_mean_squared_logarithmic_error(y_true, y_pred) -> float:
    """
    Вычисление метрики RMSLE для задачи регрессии.

    Parameters
    ----------
    y_true: array-like
        Вектор целевой переменной.

    y_pred: array-like
        Вектор прогнозов.

    Returns
    -------
    RMSLE: float
        Значение метрики RMSLE.
    
    """
    score = (np.log(y_pred + 1) - np.log(y_true + 1)) ** 2
    score = np.sum(score) / score.shape[0]
    return np.sqrt(score)


def evaluate_regression_models(model, transformer, **eval_sets):
    """
    Оценка качества модели регрессии.

    Parameters
    ----------
    model: dspl.models
        Обученная модель регрессии.

    transformer: dspl.feature_extraction.LogTargetTransformer
        Трансформер целевой переменной в логарифмическую шкалу.

    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Словарь с выборками, для которых требуется рассчитать статистику.
        Ключ словаря - название выборки (train / valid / ...), значение -
        кортеж с матрицей признаков (data) и вектором ответов (target).

    """
    for sample in eval_sets:
        data, target = eval_sets[sample]
        prediction = model.transform(data)

        if transformer.fitted:
            prediction = transformer.inverse_transform(prediction)
            target = transformer.inverse_transform(target)

        rmse = root_mean_squared_error(target, prediction)
        mae = mean_absolute_error(target, prediction)
        r2 = r2_score(target, prediction)

        msg = (f"{sample}-score: \tMAE = {round(mae, 2)}, "
               f"R2 = {round(r2, 2)}, RMSE = {round(rmse, 2)}")
        print(msg)


def select_eval_metric(config: dict) -> callable:
    """
    Выбор метрики для использования в DS-Template.
    Доступен выбор одной из метрик: MAE, MSE, RMSE, RMSLE, R2.
    Если выбирается метрика не из указанного списка, по умолчанию,
    будет использована метрика MAE.

    Parameters
    ----------
    config: dict
        Конфигурационный файл эксперимента.

    Returns
    -------
    func: callable
        Функция для вычисления метрики качества.

    """
    metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "rmsle": root_mean_squared_logarithmic_error,
        "r2": r2_score
    }
    return metrics.get(config.get("eval_metric"), "mae")
