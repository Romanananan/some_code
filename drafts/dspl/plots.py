"""
# plots.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

==========================================================================

Модуль с реализацией функций для отрисовки графиков.

Доступные сущности:
- plot_roc_curve: построение ROC-кривой.
- plot_precision_recall_curve: построение PR-AUC кривой.
- plot_mean_pred_and_target: построение кривой среднего прогноза в бакете
  и среднего ответа в бакете.
- plot_binary_graph: построение всех кривой на едином полотне.

==========================================================================

"""

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt

from .metrics import gini_score, calculate_quantile_bins


def plot_roc_curve(y_true, y_pred):
    """
    Построение графика ROC-кривой.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    plt.title("ROC-Curve", size=13)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    gini = gini_score(y_true, y_pred)
    label = "GINI: {:.4f}".format(gini)

    plt.plot(fpr, tpr, linewidth=3, label=label.format(gini), color="#534275")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.25)
    plt.legend(loc="best", fontsize=13)
    plt.xlabel("False Positive Rate (Sensitivity)", size=13)
    plt.ylabel("True Positive Rate (1 - Specificity)", size=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_precision_recall_curve(y_true, y_pred):
    """
    Построение графика для Precision-Recall кривой.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.
    """
    plt.title("Precision-Recall Curve", size=13)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    plt.plot(
        recall,
        precision,
        color="#534275",
        linewidth=3,
        label="PR-AUC:{:.4f}".format(pr_auc)
    )
    plt.axhline(np.mean(y_true), color="black", alpha=0.5, linestyle="--")
    plt.legend(loc="best", fontsize=13)
    plt.ylabel("precision", size=13)
    plt.xlabel("recall", size=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_mean_prediction_in_bin(y_true, y_pred, n_bins: int = 20):
    """
    Построение графика зависимости среднего прогноза и среднего
    значения целевой переменной в бине.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    n_bins: integer, optional, default = 20
        Количество квантильных бинов.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    bins = calculate_quantile_bins(y_pred, n_bins=n_bins)
    y_pred_mean = pd.Series(y_pred).groupby(bins).mean()
    y_true_mean = pd.Series(y_true).groupby(bins).mean()
    plt.plot(y_pred_mean.values, linewidth=3, color="#534275", label="y-pred")
    plt.plot(y_true_mean.values, linewidth=3, color="#427553", label="y-true")
    plt.xticks(ticks=range(n_bins), labels=range(0, n_bins, 1))
    plt.xlim(0, np.max(bins[bins<=n_bins]))
    plt.xlabel("bin_number", size=13)

    if y_true.nunique() <= 2:
        plt.ylabel("eventrate", size=13)
    else:
        plt.ylabel("mean-target", size=13)

        y_true_bins = pd.Series(y_true).groupby(bins)
        y_true_25p = y_true_bins.apply(
            lambda x: np.percentile(x, 25)
        )
        y_true_50p = y_true_bins.apply(
            lambda x: np.percentile(x, 50)
        )
        y_true_75p = y_true_bins.apply(
            lambda x: np.percentile(x, 75)
        )
        plt.plot(
            y_true_25p.values, label="real 25-percentile",
            color="orange", linestyle="--", alpha=0.5)
        plt.plot(
            y_true_50p.values, label="real 50-percentile",
            color="orange", linewidth=2, alpha=0.5)
        plt.plot(
            y_true_75p.values, label="real 75-percentile",
            color="orange", linestyle="--", alpha=0.5)

    plt.legend(loc="best", fontsize=13)


def plot_mean_pred_and_target(y_true, y_pred, n_bins: int = 20):
    """
    Построение графика зависимости среднего прогноза в
    бакете против среднего значения целевой метки в бакете.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    n_bins: integer, optional, default = 20
        Количество квантильных бинов.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    bins = calculate_quantile_bins(y_pred, n_bins=n_bins)
    y_pred_mean = pd.Series(y_pred).groupby(bins).mean()
    y_true_mean = pd.Series(y_true).groupby(bins).mean()
    plt.plot(y_pred_mean.values, y_true_mean.values, linewidth=3, color="#534275")
    plt.plot([0, max(y_pred_mean.values)], [0, max(y_true_mean.values)],
             color="black", alpha=0.5, linestyle="--")
    plt.xlim(min(y_pred_mean.values), max(y_pred_mean.values))
    plt.ylim(min(y_pred_mean.values), max(y_true_mean.values))
    plt.xlabel("mean-prediction", size=13)

    if y_true.nunique() <= 2:
        plt.ylabel("eventrate", size=13)
    else:
        plt.ylabel("mean-target", size=13)


def plot_scatter(y_true, y_pred):
    """
    Построение scatter-plot y_true vs y_pred.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    n_points = np.max([10000, 0.1*len(y_pred)])
    idx_points = np.random.randint(0, len(y_true), int(n_points))

    y_true_, y_pred_ = y_true[idx_points], y_pred[idx_points]
    plt.scatter(y_true_, y_pred_, alpha=0.25, color="#534275")
    plt.plot(
        [y_true_.min(), y_true_.max()],
        [y_true_.min(), y_true_.max()],
        color="orange", linestyle="--", linewidth=3)
    plt.xlim(np.percentile(y_pred, 1), np.percentile(y_pred, 99))
    plt.ylim(np.percentile(y_pred, 1), np.percentile(y_pred, 99))
    plt.ylabel("y_real", size=14)
    plt.xlabel("y_true", size=14)


def plot_binary_graph(y_true, y_pred, name: str, plot_dim=(18, 4)):
    """
    Построение графиков для бинарной классификации.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    name: string
        Имя файла для сохранения графика.

    plot_dum: Tuple[int, int], optional, default = (16, 4)
        Размер графиков.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    fig = plt.figure(figsize=plot_dim)
    plt.subplot(141)
    plot_roc_curve(y_true, y_pred)

    plt.subplot(142)
    plot_precision_recall_curve(y_true, y_pred)

    plt.subplot(143)
    try:
        plot_mean_prediction_in_bin(y_true, y_pred)
    except ValueError:
        pass
    except TypeError:
        pass

    plt.subplot(144)
    try:
        plot_mean_pred_and_target(y_true, y_pred)
    except ValueError:
        pass
    except TypeError:
        pass
    plt.savefig(f"{name}.png")


def plot_regression_graph(y_true, y_pred, name: str, plot_dim=(20, 6)):
    """
    Построение графиков для задачи регрессии.

    Parameters
    ----------
    y_true: array-like, shape = [n_samples, ]
        Вектор целевой переменной.

    y_pred: array-like, shape = [n_samples, ]
        Вектор прогнозов.

    name: string
        Имя файла для сохранения графика.

    plot_dum: Tuple[int, int], optional, default = (16, 4)
        Размер графиков.

    Returns
    -------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.

    """
    fig = plt.figure(figsize=plot_dim)
    plt.subplot(131)
    plot_scatter(y_true, y_pred)

    plt.subplot(132)
    try:
        plot_mean_prediction_in_bin(y_true, y_pred)
    except ValueError:
        pass
    except TypeError:
        pass

    plt.subplot(133)
    try:
        plot_mean_pred_and_target(y_true, y_pred)
    except ValueError:
        pass
    except TypeError:
        pass
    plt.savefig(f"{name}.png")
