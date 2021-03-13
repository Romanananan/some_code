"""
# report.py

# Team: DS.Platform (Change)
# Author: Nikita Varganov
# e-mail: Varganov.N.V@sberbank.ru

==========================================================================

Модуль с реализацией сущностей для пострения отчета о разработанных моделях.

Доступные сущности:
- ClassificationDevelopmentReport: построение отчета о разработанных 
    моделях бинарной классификации.
- RegressionDevelopmentReport: построение отчета о разработанных
    моделях регрессии.

==========================================================================

"""

from copy import deepcopy
from typing import Optional
from tqdm import tqdm

import numpy as np
import pandas as pd

from .base import BaseReport
from .base import create_used_features_stats
from .regression_metrics import CalculateRegressionMetrics
from .classification_metrics import CalculateBinaryMetrics
from .classification_metrics import CalculateDataStatistics as Binary_DS
from .classification_metrics import CalculateDetailedMetrics as Binary_DM
from .regression_metrics import CalculateDetailedMetrics as Regression_DM
from .regression_metrics import CalculateDataStatistics as Regression_DS
from ..plots import plot_binary_graph, plot_regression_graph


class ClassificationDevelopmentReport(BaseReport):
    """
    Отчет о разработанных моделях в DS-Template.

    Отчет содержит:
        - статистику по данным, которые использовались для построения
          моделей: статистика по выборкам (train / valid / ...) и
          признакам;

        - отчет об однофакторном анализе переменных (отбор
          переменных, с помощью метрики Джини);

        - сравнение построенных моделей по метрикам GINI, PR_AUC,
          Log-Loss на выборках (train / valid / ...);

        - детальные метрики для пары модель / выборка.

    Parameters
    ----------
    models: dict
        Словарь с экземплярами построенных моделей.

    saver: src.utils.INFOSaver
        pass

    config: dict
        Конфигурационный файл параметров эксперимента.

    n_bins: integer, optional, default = 20
        Количество бинов для разбиения вектора прогнозов.

    """
    def __init__(self, models, saver, config, n_bins: int = 20):
        self.models = deepcopy(models)
        self.encoder = self.models.pop("encoder")
        self.gini = self.models.pop("gini_importance")
        super().__init__(saver)

        if "psi_importance" in self.models:
            self.psi = self.models.pop("psi_importance")
        else:
            self.psi = None

        self.config = config
        self.n_bins = n_bins

    def create_first_page(self, **eval_sets):
        """
        Первая страница отчета - статистика по исследуемым данным.

        Отчет содержит:
            - статистику по выборкам, которые были использованы для
              построения / валидации / тестирования модели: название
              выборки, количество наблюдений, количество целевых
              событий и долю целевого события в выборке.

            - общую статистику по переменным: название целевой переменной,
              количество категориальных переменных, количество
              непрерывных переменных.

            - детальную статистику по каждой переменным: количество
              непропущенных значений, среднее значение переменной,
              стандартное отклонение по переменной,
              перцентили (0, 25, 50, 75, 100).

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = Binary_DS(
            self.encoder, self.gini, self.config
        )
        result = transformer.transform(**eval_sets)

        startows = [0, 2 + result[0].shape[0], 4 + result[0].shape[0] + result[1].shape[0]]
        num_formats = [10, None, None]

        for data, startrow, num_format in zip(result, startows, num_formats):
            data.to_excel(
                self.writer, startrow=startrow, sheet_name="Data_Statistics", index=False
            )
            self.set_style(data, "Data_Statistics", startrow, num_format=num_format)

        self.add_numeric_format(result[2], "Data_Statistics", startrow=startows[-1])

    def create_second_page(self, **eval_sets):
        """
        Вторая страница отчета - статистика по однофакторному
        анализу разделяющей способности переменных, измеренной
        метрикой Джини.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        self.gini.to_excel(self.writer, "GINI-Importance", index=False)
        self.set_style(self.gini, "GINI-Importance", 0)
        ws = self.sheets["GINI-Importance"]

        ws.write_string(
            "F2", "Selected - флаг, означающий включение признака в модель")
        ws.write_string(
            "F3", "Selected = 1 - признак включен в модель")
        ws.write_string(
            "F4", "Selected = 0 - признак не включен в модель")
        ws.write_string(
            "F6", "Категориальные переменные автоматически участвуют в обучении"
        )

        ws.set_column(5, 5, 62)
        gini_columns = [col for col in self.gini.columns if "GINI" in col]
        for column_number, column in enumerate(gini_columns):
            self.add_eventrate_format(
                self.gini[column], "GINI-Importance", startcol=1+column_number, fmt=2)

    def create_psi_report(self, **eval_sets):
        """
        Опциональная страница в отчете со статистикой PSI.
        Страница создается, если PSI-был рассчитан и находится в self.models.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        self.psi.to_excel(self.writer, sheet_name="PSI-Importance", index=False)
        self.set_style(self.psi, "PSI-Importance", 0)
        ws = self.sheets["PSI-Importance"]

        ws.write_string(
            "F2", "Selected - флаг, означающий включение признака в модель")
        ws.write_string(
            "F3", "Selected = 1 - признак включен в модель")
        ws.write_string(
            "F4", "Selected = 0 - признак не включен в модель")
        ws.set_column(5, 5, 62)
        self.add_eventrate_format(self.psi["PSI"], "PSI-Importance", startcol=1, fmt="0.0000")

    def create_third_page(self, **eval_sets):
        """
        Третья [четвертая] страница отчета - метрики бинарной
        классификации для каждой модели из self.models и каждая
        выборки из eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = CalculateBinaryMetrics(self.models)
        result = transformer.transform(**eval_sets)
        self.predictions = transformer.predictions_

        startcol, endcol = 2 + len(eval_sets), 2 + 3*len(eval_sets) - 1
        result.to_excel(self.writer, sheet_name="Compare Models", index=False)
        self.set_style(result, "Compare Models", 0)

        cols = [col for col in result.columns if "gini" in col]
        cols = cols + ["Название модели", "детали о модели"]
        df_a = result.drop("детали о модели", axis=1)
        df_b = result.drop(cols, axis=1)

        # серый цвет для метрик PR-AUC, Log-Loss
        self.add_text_color("Compare Models", startcol, endcol)
        self.add_numeric_format(df_a, "Compare Models", 0, min_value=100)
        self.add_numeric_format(df_b, "Compare Models", 0, 1+len(eval_sets), color="C8C8C8")

    def create_four_page(self, **eval_sets):
        """
        Четвертая [пятая] страница отчета - список используемых признаков.
        """
        df = create_used_features_stats(self.gini, self.models)
        df.to_excel(self.writer, sheet_name="Used Features", index=False)
        self.set_style(df, "Used Features", 0)

    def create_model_report(self, **eval_sets):
        """
        Страницы с отчетом для пары модель / выборка из eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = Binary_DM(self.n_bins)
        for model in tqdm(self.models, leave=False):
            for sample in eval_sets:
                sheet_name = f"{sample} {model}"
                y_true, y_pred = eval_sets[sample][1], self.predictions[model][sample]

                data = transformer.transform(y_true, y_pred)
                data.to_excel(self.writer, sheet_name=sheet_name, index=False)

                self.set_style(data, sheet_name, 0)
                self.add_numeric_format(data, sheet_name, min_value=100)
                self.add_eventrate_format(data["eventrate"], sheet_name)
                self.create_compare_models_url(data, sheet_name)
                self.create_model_url(**eval_sets)

                # график
                ws = self.sheets[sheet_name]
                plot_binary_graph(y_true, y_pred, f"{self.dir}/images/{sheet_name}")
                ws.insert_image(f"A{len(data)+5}", f"{self.dir}/images/{sheet_name}.png")

    def transform(self, **eval_sets):
        """
        Создание отчета о разработанных моделях.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        self.create_first_page(**eval_sets)
        self.create_second_page(**eval_sets)

        if isinstance(self.psi, pd.DataFrame):
            self.create_psi_report(**eval_sets)

        self.create_third_page(**eval_sets)
        self.create_four_page(**eval_sets)
        self.create_model_report(**eval_sets)

        self.writer.save()


class RegressionDevelopmentReport(BaseReport):
    """
    Отчет о разработанных моделях в DS-Template.

    Отчет содержит:
        - статистику по данным, которые использовались для построения
          моделей: статистика по выборкам (train / valid / ...) и
          признакам;

        - отчет об однофакторном анализе переменных (отбор
          переменных, с помощью метрики Джини);

        - сравнение построенных моделей по метрикам GINI, PR_AUC,
          Log-Loss на выборках (train / valid / ...);

        - детальные метрики для пары модель / выборка.

    Parameters
    ----------
    models: dict
        Словарь с экземплярами построенных моделей.

    saver: src.utils.INFOSaver
        pass

    config: dict
        Конфигурационный файл параметров эксперимента.

    n_bins: integer, optional, default = 20
        Количество бинов для разбиения вектора прогнозов.

    """
    def __init__(self, models, saver, config, n_bins: int = 20):
        self.models = deepcopy(models)
        self.encoder = self.models.pop("encoder")
        self.corr = self.models.pop("corr_importance")
        self.target_transformer = self.models.pop("log_target_transformer")
        super().__init__(saver)

        if "psi_importance" in self.models:
            self.psi = self.models.pop("psi_importance")
        else:
            self.psi = None

        self.config = config
        self.n_bins = n_bins
        print(self.models.keys())

    def create_first_page(self, **eval_sets):
        """
        Первая страница отчета - статистика по исследуемым данным.

        Отчет содержит:
            - статистику по выборкам, которые были использованы для
              построения / валидации / тестирования модели: название
              выборки, количество наблюдений, количество целевых
              событий и долю целевого события в выборке.

            - общую статистику по переменным: название целевой переменной,
              количество категориальных переменных, количество
              непрерывных переменных.

            - детальную статистику по каждой переменным: количество
              непропущенных значений, среднее значение переменной,
              стандартное отклонение по переменной,
              перцентили (0, 25, 50, 75, 100).

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = Regression_DS(
            self.encoder, self.target_transformer, self.corr, self.config
        )
        result = transformer.transform(**eval_sets)

        if len(result) < 4:
            startows = [0, 2 + result[0].shape[0], 4 + result[0].shape[0] + result[1].shape[0]]
            num_formats = [10, None, None]
        else:
            startows = [
                0,
                2 + result[0].shape[0],
                4 + result[0].shape[0] + result[1].shape[0],
                6 + result[0].shape[0] + result[1].shape[0] + result[2].shape[0]
            ]
            num_formats = [10, 10, None, None]

        for data, startrow, num_format in zip(result, startows, num_formats):
            data.to_excel(
                self.writer, startrow=startrow, sheet_name="Data_Statistics", index=False
            )
            self.set_style(data, "Data_Statistics", startrow, num_format=None)

        self.add_numeric_format(result[0], "Data_Statistics", startrow=startows[0])
        #self.add_numeric_format(result[2], "Data_Statistics", startrow=startows[-1])
        sheet_format = self.wb.add_format({"right": True, "bottom": True})

        ws = self.sheets["Data_Statistics"]
        ws.write(len(result[0]), 9, result[0].values[-1, -1], sheet_format)

    def create_second_page(self, **eval_sets):
        """
        Вторая страница отчета - статистика по однофакторному
        анализу разделяющей способности переменных, измеренной
        метрикой Джини.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        self.corr.to_excel(self.writer, "Correlation-Importance", index=False)
        self.set_style(self.corr, "Correlation-Importance", 0)
        ws = self.sheets["Correlation-Importance"]

        ws.write_string(
            "E2", "Selected - флаг, означающий включение признака в модель")
        ws.write_string(
            "E3", "Selected = 1 - признак включен в модель")
        ws.write_string(
            "E4", "Selected = 0 - признак не включен в модель")
        ws.write_string(
            "E6", "Категориальные переменные автоматически участвуют в обучении"
        )
        ws.set_column(4, 4, 62)

        if self.corr.shape[1] > 3:
            self.add_eventrate_format(
                self.corr["Correlation-Train"], "Correlation-Importance", startcol=1, fmt=2)
            self.add_eventrate_format(
                self.corr["Correlation-Valid"], "Correlation-Importance", startcol=2, fmt=2)
        else:
            self.add_eventrate_format(
                self.corr["Correlation"], "Correlation-Importance", startcol=1, fmt=2)

    def create_psi_report(self, **eval_sets):
        """
        Опциональная страница в отчете со статистикой PSI.
        Страница создается, если PSI-был рассчитан и находится в self.models.

        """
        self.psi.to_excel(self.writer, sheet_name="PSI-Importance", index=False)
        self.set_style(self.psi, "PSI-Importance", 0)
        ws = self.sheets["PSI-Importance"]

        ws.write_string(
            "E2", "Selected - флаг, означающий включение признака в модель")
        ws.write_string(
            "E3", "Selected = 1 - признак включен в модель")
        ws.write_string(
            "E4", "Selected = 0 - признак не включен в модель")
        ws.set_column(4, 4, 62)
        self.add_eventrate_format(self.psi["PSI"], "PSI-Importance", startcol=1, fmt="0.0000")

    def create_third_page(self, **eval_sets):
        """
        Третья [четвертая] страница отчета - метрики задачи
        регрессии для каждой модели из self.models и каждая
        выборки из eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = CalculateRegressionMetrics(self.target_transformer, self.models)
        result = transformer.transform(**eval_sets)
        self.predictions = transformer.predictions_

        startcol, endcol = 2 + len(eval_sets), 2 + 3*len(eval_sets) - 1
        result.to_excel(self.writer, sheet_name="Compare Models", index=False)
        self.set_style(result, "Compare Models", 0)

        cols = [col for col in result.columns if "MAE" in col]
        cols = cols + ["Название модели", "детали о модели"]
        df_a = result.drop("детали о модели", axis=1)
        df_b = result.drop(cols, axis=1)

        # серый цвет для метрик PR-AUC, Log-Loss
        self.add_text_color("Compare Models", startcol, endcol)
        self.add_numeric_format(df_a, "Compare Models", 0, min_value=100)
        self.add_numeric_format(df_b, "Compare Models", 0, 1+len(eval_sets), color="C8C8C8")

    def create_model_report(self, **eval_sets):
        """
        Страницы с отчетом для пары модель / выборка из eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        transformer = Regression_DM(self.target_transformer, self.n_bins)
        for model in tqdm(self.models, leave=False):
            for sample in eval_sets:
                sheet_name = f"{sample} {model}"
                y_true, y_pred = eval_sets[sample][1], self.predictions[model][sample]

                if self.target_transformer.fitted:
                    y_true = self.target_transformer.inverse_transform(y_true)

                print(np.mean(y_true), np.mean(y_pred))
                data = transformer.transform(y_true, y_pred)
                data.to_excel(self.writer, sheet_name=sheet_name, index=False)

                self.set_style(data, sheet_name, 0)
                self.add_numeric_format(data, sheet_name, min_value=100)
                self.create_compare_models_url(data, sheet_name)
                self.create_model_url(**eval_sets)

                # график
                ws = self.sheets[sheet_name]
                plot_regression_graph(y_true, y_pred, f"{self.dir}/images/{sheet_name}")
                ws.insert_image(f"A{len(data)+5}", f"{self.dir}/images/{sheet_name}.png")

    def create_four_page(self, **eval_sets):
        """
        Четвертая [пятая] страница отчета - список используемых признаков.
        """
        df = create_used_features_stats(self.corr, self.models)
        df.to_excel(self.writer, sheet_name="Used Features", index=False)
        self.set_style(df, "Used Features", 0)

    def create_model_url(self, **eval_sets):
        """
        Создание ссылки на лист с отчетом модель / выборка и
        добавление на лист Compare Models.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        ws = self.sheets["Compare Models"]
        cols = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        sheet_format = self.wb.add_format({"right": True})
        train_sheets = [sheet for sheet in self.sheets if "train" in sheet]

        for sheet_number, sheet_name in enumerate(train_sheets):
            url = f"internal:'{sheet_name}'!A1"
            string = f"Ссылка на лист {sheet_name}"
            try:
                cell_name = cols[2 + 6*len(eval_sets)]
            except IndexError:
                n_letter = abs(len(cols) - 6*len(eval_sets) - 2)
                cell_name = "A{}".format(cols[n_letter])

            ws.write_url(f"{cell_name}{sheet_number + 2}", url, sheet_format, string)

        sheet_format = self.wb.add_format({"right": True, "bottom": True})
        ws.write_url(f"{cell_name}{sheet_number + 2}", url, sheet_format, string)

    def transform(self, **eval_sets):
        """
        Создание отчета о разработанных моделях.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        self.create_first_page(**eval_sets)
        self.create_second_page(**eval_sets)

        if isinstance(self.psi, pd.DataFrame):
            self.create_psi_report(**eval_sets)

        self.create_third_page(**eval_sets)
        self.create_four_page(**eval_sets)
        self.create_model_report(**eval_sets)
        self.writer.save()
