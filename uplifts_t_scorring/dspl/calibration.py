"""
===============================================================================
calibration.py

Модуль с реализацией калибровки калибровок моделей бинарной классификации из
шаблоа ds_templates или поддерживающих этот API.


Сущности:
 - create_calib_stats     - расчет статистик по калибровке
 - plot_calibration_curve - построение клаибровочной кривой y_true-y_pred
 - plot_bin_curve         - построение кривой по бинам прогноза

 - CalibrationReport      - класс создания отчета по калибровкам

 - IsotonicCalibration    - реализация калибровки изотонической регрессией
 - LogisticCalibration    - рализация калибровки логистической регрессий
 - LinearCalibration      - реализация калибровки линейной регрессией
 - DecisionTreeCalibration - калибровка решающим деревом и лин. моделями

 - Calibration            -  интерфейсный класс для выполнения калибровки
===============================================================================
"""
import pandas as pd
import numpy as np
import pickle
import os
from contextlib import suppress

from typing import Callable, Tuple

import matplotlib.pyplot as plt

from copy import deepcopy
from scipy.special import logit, expit

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import brier_score_loss, log_loss, \
    mean_absolute_error, mean_squared_error

from dspl.validation_report.FormatExcelWriter import FormatExcelWriter
from dspl.metrics import calculate_quantile_bins

def create_calib_stats(df: pd.DataFrame):
    """
    Функция формирования датафрейма со статистикой по калибровке на
    конкретной выборке в разрезе бинов прогноза

    Parameters:
    ----------
    df: pd.DataFrame
        Датафрейм с 3-мя колонками:
            - y_pred  - вектор прогнозов модели
            - y_calib - вектор прогноза калибровки
            - y_true  - вектор истинных значений


    Return:
    -------
    pd.DataFrame
        Статистики по калибровке в разрезе бинов прогноза исходной модели
        Столбцы:
            - bin:                номер бина
            - mean proba:         средний прогноз модели в бине
            - calibration proba:  средний прогноз калибровки в бине
            - event rate:         доля целевых событий в бине
            - # obs:              количество наблюдений в бине
            - MAE:                взв. mean absolute error по прогнозу в бине
            - MAE calibrated:     взв. mean absolute error по калибровке в бине
            - Brier:              MSE прогноза на всех наблюдениях
            - Brier calibrated:   MSE калибровки на всех наблюдениях
            - logloss:            logloss прогнозана всех наблюдениях
            - logloss calibrated: logloss калибровки на всех наблюдениях
    """
    # stats
    df["bin"] = calculate_quantile_bins(df["y_pred"], n_bins=21)

    df_group = df.groupby(by="bin")
    stats = df_group.agg({"y_pred": ["mean"],
                          "y_calib": ["mean"],
                          "y_true": ["mean", "count"]
                          })

    stats.columns = ["mean proba", "calibration proba", "event rate", "# obs"]

    # metrics:

    # expected calibration error = weighted mean absolute error
    mae =  mean_absolute_error(stats["event rate"], stats["mean proba"],
                               sample_weight=stats["# obs"])
    mae_calib = mean_absolute_error(stats["event rate"],
                                    stats["calibration proba"],
                                    sample_weight=stats["# obs"])
    stats.loc["Total", "MAE"] = mae
    stats.loc["Total", "MAE calibrated"] = mae_calib

    # mean square error = brier score
    stats.loc["Total", "Brier"] = mean_squared_error(df["y_true"], df["y_pred"])
    stats.loc["Total", "Brier calibrated"] = mean_squared_error(df["y_true"],
                                                                df["y_calib"])

    #logloss
    stats.loc["Total", "logloss"] = log_loss(df["y_true"], df["y_pred"],
                                             eps=1e-5)
    stats.loc["Total", "logloss calibrated"] = log_loss(df["y_true"],
                                                        df["y_calib"],
                                                        eps=1e-5)


    # total row
    stats.loc["Total", "mean proba"] = df["y_pred"].mean()
    stats.loc["Total", "calibration proba"] = df["y_calib"].mean()
    stats.loc["Total", "event rate"] = df["y_true"].mean()
    stats.loc["Total", "# obs"] = stats["# obs"].sum()

    stats.insert(loc=0, column="bin", value=stats.index)

    return stats.fillna(".")


def plot_calibration_curve(pred_df: pd.DataFrame, title:str=None):
    """
    Функция построения графика для диагностирования необхдоимости калибровки:
    строит зависимость y_pred - y_true, разбив исходные векторы на бины
    Отрисовывает 2 графика,
     y_calibrated - y_true
     y_pred       - y_true

    Parameters:
    ----------
    df: pd.DataFrame
        Датафрейм с 3-мя колонками:
            - y_pred  - вектор прогнозов модели
            - y_calib - вектор прогноза калибровки
            - y_true  - вектор истинных значений

    """
    # порезать на бакеты
    pred_df["bin"] = calculate_quantile_bins(pred_df["y_pred"], 21)

    pred_df_grouped = pred_df.groupby(by="bin").mean()

    # plt.figure(figsize=(12,8))
    plt.plot(pred_df_grouped["y_pred"], pred_df_grouped["y_true"], marker='o',
             label="model", linewidth=3)
    plt.plot(pred_df_grouped["y_calib"], pred_df_grouped["y_true"], marker='o',
             label="model calibrated", linewidth=4)
    xlim = ylim = pred_df_grouped["y_true"].max()
    plt.plot([0, xlim],
             [0, ylim], "k--")
    plt.grid()
    plt.xlabel("mean prediction")
    plt.ylabel("mean target")
    plt.legend()
    if title is not None:
        plt.title(title)


def plot_bin_curve(pred_df: pd.DataFrame, title: str=None):
    """
    Функция построения графика среднего прогноза в бинах
    Отрисовывает 3 графика,
        bin - event_rate
        bin - mean proba
        bin - mean calibration

    Parameters:
    ----------
    df: pd.DataFrame
        Датафрейм с 3-мя колонками:
            - y_pred  - вектор прогнозов модели
            - y_calib - вектор прогноза калибровки
            - y_true  - вектор истинных значений

    """
    # порезать на бакеты
    pred_df["bin"] = calculate_quantile_bins(pred_df["y_pred"], 21)

    pred_df_grouped = pred_df.groupby(by="bin").mean()

    # plt.figure(figsize=(12,8))
    plt.plot(pred_df_grouped["y_true"], "green", marker='o', label="y_true",
             linewidth=3)
    plt.plot(pred_df_grouped["y_pred"], marker='o', label="y_pred",
             linewidth=3)
    plt.plot(pred_df_grouped["y_calib"], marker='o', label="y_calibrated",
             linewidth=4)
    plt.grid()
    plt.xlabel("bin")
    plt.ylabel("mean prediction")
    plt.legend()
    if title is not None:
        plt.title(title)


class CalibrationReport:
    """
    Класс реализациия создания и сохранения отчета по калибровке модели

    Parameters:
    ----------
    calibrators: dict {<Имя калибровки>: <объект класса Calibration>}
        Словарь со всеми калибровками по которым нужно построить отчет

    config: dict
        словарь с параметрами запуска

    """

    def __init__(self, calibrations: dict, config: dict):

        self.calibrators = deepcopy(calibrations)
        self.reports = {}
        self.make_dirs(config)

    def make_dirs(self, config: dict):
        """
        Создание директории с префиксом calib_ в каталоге, где лежит
        калибруемая модель

        Parameters
        ----------

        config: dict
            Словарь с параметрами калибруемое модели, обязательно должны быть
            параметры "model_name" и "run_number" из шаблона построения модели
            бинарной класификации ds_templates

        """

        model_name = config.get("model_name", None)
        run_num = config.get("run_number", None)
        model_path = config.get("model_path", None)

        if model_name is not None:
            self.save_path = f"runs/{run_num}/models/calib_{model_name}/"

        else:
            model_name = model_path.split("/")[-1]
            path = "/".join(model_path.split("/")[:-1])

            self.save_path = f"{path}/calib_{model_name}/"

        with suppress(FileExistsError):
            os.mkdir(self.save_path)

    def _to_excel(self, df: pd.DataFrame, sheet_name: str, formats: dict = None,
                  plot: bool=False):
        """
        Метод для записи произвольного датафрейма на лист excel книги

        Parameters
        ---------
        df: pd.DataFrame
            Датафрейм, который будет записат в excel книгу

        sheet_name: str
            имя страницы для записи

        formats: dict
            словарь с перечнем форматов для столбцов датафрейма

        plot: bool
            флаг - вставлять на страницу рисунок с именем {sheet_name}.png
            из  каталога self.save_path
        """
        # write table
        formatWriter = FormatExcelWriter(self.writer)
        formatWriter.write_data_frame(df=df,
                                      pos=(0, 0),
                                      sheet=sheet_name,
                                      formats=formats)

        # insert plot
        if plot:
            sheet = self.writer.sheets[sheet_name]
            fig_path = f"{self.save_path}/{sheet_name}"
            sheet.insert_image(f"A{df.shape[0] + 4}", f"{fig_path}.png")

    @staticmethod
    def plot_calib_curves(df: pd.DataFrame, plot_name: str=None):
        """
        Метод для отрисовки и сохранения в файл двух графиков:
        calibration_curve и bin_curve

        Parameters
        ----------
        df: pd.DataFrame
            Датафрейм на основе которого будут построены графики
            Обязательные столбцы:
                - y_pred  - вектор прогнозов модели
                - y_calib - вектор прогноза калибровки
                - y_true  - вектор истинных значений

        plot_name: str
            имя для сохранения изображения в файл
        """
        plt.figure(figsize=(24, 6))
        plt.subplot(1, 2, 1)
        plot_bin_curve(df)
        plt.subplot(1, 2, 2)
        plot_calibration_curve(df)
        if plot_name is not None:
            plt.savefig(f"{plot_name}.png", bbox_inches="tight")

    def create_report(self, model: Tuple, **kwargs):
        """
        Создание отчетов по калибровкам на каждой выборке, сохранение объектов
        с калибровками

        Parameters
        ----------
        model: Tuple(<str>,<объект Calibration>)
            Кортеж с наименованием калибровки и объектом, который калибрует

        kwargs: dict{}
            словарь с keyword аргументами - выборкам, на которых необходимо
            создавать расчеты

        """
        cal_name, calibration = model

        for ds_name, (X, y) in kwargs.items():
            pred_df = pd.DataFrame({"y_true": y,
                                    "y_pred": calibration.get_y_pred(X),
                                    "y_calib": calibration.transform(X)
                                    })

            report = create_calib_stats(pred_df)
            self.reports[f"{cal_name}_{ds_name}"] = report

            self.plot_calib_curves(pred_df, f"{self.save_path}/"
                                            f"{cal_name}_{ds_name}")

            with open(f"{self.save_path}/{cal_name}.pkl", "wb") as mod:
                pickle.dump(calibration, mod)

    def print_reports(self, **kwargs):
        """
        Вывод на страницы excel книги отчетов по каждой калибровк на всех
        имеющихся выборках
        """
        int_number = "## ##0"
        float_number_high = "## ##0.00"
        float_number_low = "## ##0.00000"

        # Кастомный формат для таблицы
        format = {"num_format": {
            int_number: ["bin", "#obs"],
            float_number_high: ["mean proba", "calibration proba",
                                        "event rate"],
            float_number_low: ["MAE", "MAE calibrated",	"Brier",
                                       "Brier calibrated",	"logloss",
                                       "logloss calibrated"]
            }
        }
        for calib_name in self.calibrators.keys():
            for ds_name in kwargs.keys():
                report = self.reports[f"{calib_name}_{ds_name}"]
                self._to_excel(report, f"{calib_name}_{ds_name}", format, True)

    def crete_comparison(self, **kwargs):

        # формат для печати в excel
        float_percentage = "0.00%"
        float_number_low = "## ##0.00000"

        format = { "num_format":{
            float_number_low: ['MAE_train', 'MAE_calibrated_train', 'MAE_valid',
                      'MAE_calibrated_valid', 'MAE_OOT', 'MAE_calibrated_OOT',
                      'Brier_train', 'Brier_calibrated_train', 'Brier_valid',
                      'Brier_calibrated_valid', 'Brier_OOT',
                      'Brier_calibrated_OOT','logloss_train',
                      'logloss_calibrated_train','logloss_valid',
                      'logloss_calibrated_valid','logloss_OOT',
                      'logloss_calibrated_OOT'],
            float_percentage: ['delta MAE_train', 'delta Brier_train',
                               'delta logloss_train', 'delta MAE_valid',
                               'delta Brier_valid', 'delta logloss_valid',
                               'delta MAE_OOT', 'delta Brier_OOT',
                               'delta logloss_OOT']
            }
        }

        plt.figure(figsize=(35, 30))
        summary = pd.DataFrame(index=list(self.calibrators.keys()))

        # comparison table
        ds_list = list(kwargs.keys())
        for calib_name, calibrator in self.calibrators.items():

            # weighted MAE
            for line_num, ds_name in enumerate(ds_list):
                report = self.reports[f"{calib_name}_{ds_name}"]
                summary.loc[calib_name,f"MAE_{ds_name}"] = report.loc["Total",
                    "MAE"]
                summary.loc[calib_name, f"MAE_calibrated_{ds_name}"] = \
                    report.loc["Total", "MAE calibrated"]

            # brier score = mse for classification
            for ds_name in ds_list:
                report = self.reports[f"{calib_name}_{ds_name}"]
                summary.loc[calib_name, f"Brier_{ds_name}"] = report.loc[
                    "Total", "Brier"]
                summary.loc[calib_name, f"Brier_calibrated_{ds_name}"] = \
                    report.loc["Total", "Brier calibrated"]
            # logloss

            for ds_name in ds_list:
                report = self.reports[f"{calib_name}_{ds_name}"]
                summary.loc[calib_name, f"logloss_{ds_name}"] = report.loc[
                 "Total", "logloss"]
                summary.loc[calib_name, f"logloss_calibrated_{ds_name}"] = \
                    report.loc["Total", "logloss calibrated"]

            for ds_name in ds_list:
                # deltas
                # delta ECE
                summary[f"delta MAE_{ds_name}"] = \
                    summary[f"MAE_calibrated_{ds_name}"] - \
                    summary[f"MAE_{ds_name}"]

                # delta Brier
                summary[f"delta Brier_{ds_name}"] = \
                    summary[f"Brier_calibrated_{ds_name}"] - \
                    summary[f"Brier_{ds_name}"]

                # delta logloss
                summary[f"delta logloss_{ds_name}"] = \
                    summary[f"logloss_calibrated_{ds_name}"] - \
                    summary[f"logloss_{ds_name}"]

        # comparison plots
        plot_lines = len(kwargs)
        subplot_pos = 1
        for ds_name, (x,y) in kwargs.items():
            for calib_name, calibrator in self.calibrators.items():
                # add subplot
                plt.subplot(2*plot_lines, 7, subplot_pos)
                pred_df = pd.DataFrame({"y_true": y,
                                        "y_pred": calibrator.get_y_pred(x),
                                        "y_calib": calibrator.transform(x)
                })
                plot_calibration_curve(pred_df, f"{calib_name}_{ds_name}")
                plt.subplot(2*plot_lines, 7, subplot_pos+plot_lines*7)
                plot_bin_curve(pred_df, f"{calib_name}_{ds_name}")

                subplot_pos += 1

        # save figure
        plt.savefig(f"{self.save_path}/calibration_comparison.png",
                    bbox_index="tight")
        # reset index
        summary.insert(loc=0, column="calibration", value=summary.index)
        desc = [
                "Линейная регрессия на бинах прогноза",
                "Линейная регрессия на шансах в бинах прогноза",
                "Линейная регрессия на логарифме шансов в бинах прогноза",
                "Логистическая регрессия на всех наблюдениях",
                "Логистическая регрессия на шансах прогнозов наблюдений",
                "Логистическая регрессия на логарифме шансов прогнозов "
                "наблюдений",
                "Изотоническая регрессия"
        ]
        summary.insert(loc=1, column="description", value=desc)

        self._to_excel(summary, sheet_name="calibration_comparison",
                       formats=format, plot=True)

    def print_equations(self):
        for name, calibration in self.calibrators.items():
            if name in ["linear", "logit"]:
                print(f"{name}: {calibration.get_equation()}")

    def create_equations(self):
        equations = pd.DataFrame(columns=["equation"])
        for name, calib in self.calibrators.items():
            if hasattr(calib, "get_equation"):
                equations.loc[name] = calib.get_equation()

        equations = equations.reset_index()
        self._to_excel(equations, "equations", None,  False)

    def create_data_stats(self, **kwargs):
        """
        Построение отчета о данных, которые использованы
        для обучения / валидации модели.

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        data_dict = {}
        for data_name, (x, y) in kwargs.items():
            data_dict[data_name] = [x.shape[0], np.sum(y), np.mean(y)]
        data_stats = pd.DataFrame(data_dict).T
        data_stats = data_stats.reset_index()
        data_stats.columns = ["выборка", "# наблюдений", "# events",
                                  "# eventrate"]

        # стандартные форматы чисел
        int_number = "## ##0"
        float_percentage = "0.00%"
        format = {"num_format": {int_number: ["# наблюдений", "# events"],
                                  float_percentage: ["# eventrate"]}
                   }
        self._to_excel(data_stats, "Data sets", format)

    def transform(self, **kwargs):
        self.writer = pd.ExcelWriter(path=f"{self.save_path}/calibration.xlsx")

        self.create_data_stats(**kwargs)

        for model in self.calibrators.items():
            self.create_report(model, **kwargs)

        self.crete_comparison(**kwargs)
        self.create_equations()
        self.print_reports(**kwargs)
        self.writer.save()


class IsotonicCalibration(BaseEstimator, TransformerMixin):
    """
    Построение модели изотонической регресии на наблюдениях:
    y_pred -> y_target
    """

    def __init__(self):
        self.calibration = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_pred: pd.Series, y_true: pd.Series):
        self.calibration.fit(y_pred, y_true)
        return self

    def transform(self, y_pred):
        return self.calibration.transform(y_pred)


class LogisticCalibration(BaseEstimator, TransformerMixin):
    """
    Построение модели логистической регресии
    y_pred -> y_target
    """

    def __init__(self, is_logit=False, is_odds=False):
        self.calibration = LogisticRegression()
        self.is_logit = is_logit
        self.is_odds = is_odds

    def _fit_odds(self, y_pred: pd.Series, y_true: pd.Series):
        x = np.array(y_pred/(1-y_pred)).reshape(-1, 1)
        self.calibration.fit(x, y_true)

    def _fit_logit(self, y_pred: pd.Series, y_true: pd.Series):
        x = logit(np.array(y_pred).reshape(-1, 1))
        self.calibration.fit(x, y_true)

    def _fit_logreg(self, y_pred: pd.Series, y_true: pd.Series):
        x = np.array(y_pred).reshape(-1, 1)
        self.calibration.fit(x, y_true)

    def fit(self, y_pred: pd.Series, y_true: pd.Series):

        if self.is_odds:
            self._fit_odds(y_pred, y_true)
        elif self.is_logit:
            self._fit_logit(y_pred, y_true)
        else:
            self._fit_logreg(y_pred, y_true)

        return self

    def get_equation(self):
        k = float(self.calibration.coef_)
        b = float(self.calibration.intercept_)

        if self.is_odds:
            return f"1/(1+ exp(-{k}*(x/1-x) + {b}))"
        elif self.is_logit:
            return f"1/(1+ exp(-{k}*ln(x/1-x) + {b}))"
        else:
            return f"1/(1+exp(-{k}*x + {b}))"

    def transform(self, y_pred):
        if self.is_odds:
            x = np.array(y_pred / (1 - y_pred)).reshape(-1, 1)
        elif self.is_logit:
            x = logit(np.array(y_pred).reshape(-1, 1))
        else:
            x = np.array(y_pred).reshape(-1, 1)

        return self.calibration.predict_proba(x)[:, 1]


class LinearCalibration(BaseEstimator, TransformerMixin):
    """
    Построение модели линейной регресии на средних из бинов прогноза
    y_bin_mean_prediction -> y_bin_mean_target
    """

    def __init__(self, is_weighted: bool = False, is_logit: bool = False,
                 is_odds=False):
        self.calibration = LinearRegression()
        self.is_weighted = is_weighted
        self.is_logit = is_logit
        self.is_odds = is_odds

    def fit(self, y_pred, y_true):

        # данные для обучения
        pred_df = pd.DataFrame({"y_true": y_true,
                                "y_pred": y_pred})
        pred_df["pred_bin"] = calculate_quantile_bins(y_pred, 20)
        pred_df_grouped = pred_df.groupby(by="pred_bin").\
            agg({"y_pred": "mean", "y_true": ["mean", "sum"]})

        pred_df_grouped.columns = ["y_pred", "y_true", "#events"]
        pred_df_grouped["events_share"] = pred_df_grouped["#events"] \
                                        / pred_df_grouped["#events"].sum()

        x = np.array(pred_df_grouped["y_pred"]).reshape(-1, 1)
        y = pred_df_grouped["y_true"]

        # запомнить средний ER в бине прогноза - для взвешивания
        weights = pred_df_grouped["events_share"]

        if self.is_odds:
            x = np.array(x / (1 - x))
            y = np.array(y / (1 - y))

        if self.is_logit:
            x = logit(x)
            y = logit(y)

        if self.is_weighted:
            self.calibration.fit(x, y, sample_weight=weights)
        else:
            self.calibration.fit(x, y)

        return self

    def get_equation(self):
        k = float(self.calibration.coef_)
        b = float(self.calibration.intercept_)

        if self.is_odds:
            return f"y_odds = {k}*(x/(1-x)) + {b}"
        elif self.is_logit:
            return f"y_ln_odds = {k}*ln(x/1-x) + {b}"
        else:
            return f"y = {k}*x + {b}"

    def transform(self, y_pred):
        x = np.array(y_pred).reshape(-1, 1)

        if self.is_logit:
            x = logit(x)
            pred = self.calibration.predict(x)
            pred = expit(pred)

        elif self.is_odds:
            x = (x/(1-x))
            pred = self.calibration.predict(x)
            pred = (pred/(pred+1))
        else:
            pred = self.calibration.predict(x)

        return pred


class DecisionTreeCalibration(BaseEstimator, TransformerMixin):
    """
    Выполнение калибровки решеающим деревом и линейными моделями в листах
    """

    def __init__(self, model, tree_max_depth=3, rs=17):
        self.model = model
        self.rs = rs
        self.dt_calib = DecisionTreeClassifier(max_depth=tree_max_depth,
                                               random_state=rs)
        self.logits = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):

        # Обучить дерево решений
        self.dt_calib.fit(X[self.model.used_features], y)
        leafs = self.dt_calib.apply(X[self.model.used_features])

        # Обучить логистическую регрессию для каждого листа
        for leaf in np.unique(leafs):
            lr = LogisticRegression(random_state=self.rs)

            X_sub = X[leafs == leaf]
            y_pred_sub = self.model.transform(X_sub)
            y_sub = y[leafs == leaf]

            lr.fit(y_pred_sub.reshape(-1, 1), y_sub)
            self.logits[leaf] = lr

    def transform(self, X:pd.DataFrame):

        pred_df = pd.DataFrame({"y_pred": self.model.transform(X),
                                "leaf": self.dt_calib.apply(
                                    X[self.model.used_features])},
                               index=X.index)

        y_calib = pd.Series()

        # для каждого листа применить свой логит
        for lf in np.unique(pred_df.leaf):
            idx_sub = pred_df[pred_df.leaf == lf].index
            y_pred_sub = np.array(pred_df[pred_df.leaf == lf].y_pred).reshape(
                -1, 1)

            y_calib_sub = pd.Series(
                self.logits[lf].predict_proba(y_pred_sub)[:, 1],
                index=idx_sub)

            y_calib = y_calib.append(y_calib_sub)
        return y_calib


class Calibration(BaseEstimator, TransformerMixin):
    """
    Класс-интерфейс для доступа к реализациям калибровки по единому api

    """
    def __init__(self, model: Callable, method: str = "isotonic",
                 is_weighted: bool = False, features_list: list = None):
        if hasattr(model, "predict") or hasattr(model, "predict_proba") or \
                hasattr(model, "transform"):
            self.model = model
            self.method = method
            self.is_weighted = is_weighted
            self.features_list = features_list
        else:
            raise AttributeError("Model object must support prediction API via"
                                 " one of the methods: 'predict',"
                                 " 'predict_proba' or 'transform'")

    def get_y_pred(self, x: pd.DataFrame) -> pd.Series:

        if hasattr(self.model, "transform"):
            y_pred = self.model.transform(x)

        elif hasattr(self.model, "predict_proba"):
            if self.model.__module__.split(".")[0] == "dspl":
                y_pred = self.model.predict_proba(x)[:, 1]
            else:
                y_pred = self.model.predict_proba(x[self.features_list])[:, 1]

        elif hasattr(self.model, "predict"):
            if self.model.__module__.split(".")[0] == "dspl":
                y_pred = self.model.predict(x)
            else:
                y_pred = self.model.predict(x[self.features_list])

        else:
            raise AttributeError("Model object must support prediction API via"
                                 " one of the methods: 'predict',"
                                 " 'predict_proba' or 'transform'")
        return y_pred

    def fit(self, x: pd.DataFrame, y: pd.Series):

        y_pred = self.get_y_pred(x)

        # изотоническая регрессия на наблюдениях
        if self.method == "isotonic":
            self.calibrator = IsotonicCalibration()
            self.calibrator.fit(y_pred, y)

        # логистическая регрессия на наблюдениях
        elif self.method == "logistic":
            # обучение логистической регрессии на наблюдениях
            self.calibrator = LogisticCalibration()
            self.calibrator.fit(y_pred, y)

        # линейная регрессия на бакетах
        elif self.method == "linear":
            self.calibrator = LinearCalibration(is_weighted=self.is_weighted)
            self.calibrator.fit(y_pred, y)

        # линейная регрессия на шансах
        elif self.method == "linear-odds":
            self.calibrator = LinearCalibration(is_odds=True,
                                                is_weighted=self.is_weighted)
            self.calibrator.fit(y_pred, y)

        # линейная регрессия на логарифме шансов
        elif self.method == "linear-ln-odds":
            self.calibrator = LinearCalibration(is_logit=True,
                                                is_weighted=self.is_weighted)
            self.calibrator.fit(y_pred, y)

        # логистическая регрессия на шансах
        elif self.method == "logistic-odds":
            self.calibrator = LogisticCalibration(is_odds=True)
            self.calibrator.fit(y_pred, y)

        # логистическая регрессия на логарифме шансов
        elif self.method == "logistic-ln-odds":
            self.calibrator = LogisticCalibration(is_logit=True)
            self.calibrator.fit(y_pred, y)

        elif self.method == "dtree":
            self.calibrator = DecisionTreeCalibration(self.model,)
            self.calibrator.fit(x, y)

    def get_equation(self):
        if hasattr(self.calibrator, "get_equation"):
            return self.calibrator.get_equation()

    def evaluate(self, **kwargs):
        for ds_name, (x,y) in kwargs.items():
            y_pred = self.get_y_pred(x)
            y_calibrated = self.calibrator.transform(y_pred)
            brier = mean_squared_error(y, y_pred)
            brier_calib = mean_squared_error(y, y_calibrated)
            logloss = log_loss(y, y_pred, eps=1e-5)
            logloss_calib = log_loss(y, y_pred, eps=1e-5)

            print(f"{ds_name} \t Brier: {round(brier,8)} \t "
                  f"Brier calibrated: {round(brier_calib,8)} ")
            print(f"{ds_name} \t logloss: {round(logloss, 8)} \t"
                  f" logloss calibrated: {round(logloss_calib, 8)} ")

    def transform(self, x: pd.DataFrame):

        y_pred = self.get_y_pred(x)
        y_pred_calib = self.calibrator.transform(y_pred)

        return y_pred_calib