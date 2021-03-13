import pandas as pd
import numpy as np

from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df

from ..feature_extraction.transformers import DecisionTreeFeatureImportance
from ..metrics import root_mean_squared_error, mean_absolute_percentage_error
from ..utils import calculate_time_execute

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, r2_score


class RegressionMetricsChecker(Checker):
    """
    Класс реализации проверки стабильности ранжирующей способности
    модели регресии по метрикам :
        - MAE
        - MAPE
        - RMSE
        - R2
        - pearson
        - spearman

    Parameters:
    ----------
    writer-: pd.ExcelWriter
        Объект класса excel-writer для записи отчета (файл для отчета должен
        быть создан предварительно)

    model_name: str
        Имя модели для отображения в названи файлов

    model
        Объект scikit-learn like обученной модели

    features_list:list
        Список фичей, которые использует модель

    n_bins: int
        количество бинов для разбиение данных по прогнозу целевой переменной

    cat_features: list
        Список категориальных признаков

    drop_features: list
        Список мусорных признаков для исключения из анализа

    current_path: str
        Путь к рабочей директории для сохранения изображений и файла с отчетом
    """

    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 cat_features: list,
                 drop_features: list,
                 target_transformer=None,
                 n_bins: int = 20,
                 current_path=None):

        self.writer = writer
        self.model_name = model_name
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model = model
        self.target_transformer = target_transformer
        self.n_bins = n_bins
        self.current_path = current_path

    def _corr_to_excel(self, df: pd.DataFrame, sheet_name: str, pos=(0,0),
                  formats: dict = None) -> None:
        """
        Функция записи датафрейма в excel файл на указанный лист и позицию

        Parameters:
        ----------
        df: pd.DataFrame
            Датафрейм для записи в файл
        sheet_name: str
            Имя листа, на который осуществить запись
        plot: bool
            Флаг необходимости добавить на страницу с отчетом график из файла
        """

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df, pos, sheet_name, formats=formats)
        sheet = self.writer.sheets[sheet_name]

        # apply conditional format to highlight validation_report test results

        if "delta_train_vs_test" in df.columns:
            excelWriter.set_col_cond_format(df, pos, "delta_train_vs_test",
                                            upper=-0.3, lower=-0.2,
                                            order="reverse")
        if "delta_train_vs_OOT" in df.columns:
            excelWriter.set_col_cond_format(df, pos, "delta_train_vs_OOT",
                                            upper=-0.25, lower=-0.15,
                                            order="reverse")
        if "delta_OOT_vs_test" in df.columns:
            excelWriter.set_col_cond_format(df, pos, "delta_OOT_vs_test",
                                            upper=-0.2, lower=-0.15,
                                            order="reverse")

    def _model_to_excel(self, df: pd.DataFrame, sheet_name: str, pos=(0, 0),
                        formats: dict = None) -> None:
        """
        Функция записи датафрейма в excel файл на указанный лист и позицию

        Parameters:
        ----------
        df: pd.DataFrame
            Датафрейм для записи в файл
        sheet_name: str
            Имя листа, на который осуществить запись
        plot: bool
            Флаг необходимости добавить на страницу с отчетом график из файла
        """

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df, pos, sheet_name, formats=formats)
        sheet = self.writer.sheets[sheet_name]

        # apply conditional format to highlight validation_report test results

        if "delta_train_vs_test" in df.columns:
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=0, upper=0.3, lower=0.2,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=1, upper=0.3, lower=0.2,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=2, upper=0.3, lower=0.2,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=3, upper=-0.3, lower=-0.2,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=4, upper=-0.3, lower=-0.2,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_test",
                                             row_num=5, upper=-0.3, lower=-0.2,
                                             order="reverse")
            #excelWriter.set_col_cond_format(df, pos, "delta_train_vs_test",
            #                                upper=0.3, lower=0.2,
            #                                order="straight")
        if "delta_train_vs_OOT" in df.columns:
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=0, upper=0.25, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=1, upper=0.25, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=2, upper=0.25, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=3, upper=-0.25, lower=-0.15,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=4, upper=-0.25, lower=-0.15,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_train_vs_OOT",
                                             row_num=5, upper=-0.25, lower=-0.15,
                                             order="reverse")
            #excelWriter.set_col_cond_format(df, pos, "delta_train_vs_OOT",
            #                                upper=0.25, lower=0.15,
            #                                order="straight")
        if "delta_OOT_vs_test" in df.columns:
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=0, upper=0.2, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=1, upper=0.2, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=2, upper=0.2, lower=0.15,
                                             order="straight")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=3, upper=-0.2, lower=-0.15,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=4, upper=-0.2, lower=-0.15,
                                             order="reverse")
            excelWriter.set_cell_cond_format(df=df, pos=pos,
                                             col_name="delta_OOT_vs_test",
                                             row_num=5, upper=-0.2, lower=-0.15,
                                             order="reverse")
            #excelWriter.set_col_cond_format(df, pos, "delta_OOT_vs_test",
            #                                upper=0.2, lower=0.15,
            #                                order="straight")


        # Описание теста
        sheet.write_string(
            "K3", "<-- Метрики качества по построенной модели регрессии")

        sheet.write_string(
            "K10", "<-- Изменение коэффициента корреляции между значением "
                   "целевой переменной и прогнозом дерева решений, "
                   "построенном на одном признаке")
    @staticmethod
    def mean_absolute_percentage_error(y_true: pd.Series,
                                       y_pred: pd.Series) -> float:
        """
        Фукция расчитывает MAPE для прогноза и истинных
        значений целевой переменной

        Если в целевом векторе имеется значение 0 - вызовет exception

        Parameters:
        -----------
        y_true: pd.Series
            Вектор значений целевой переменной
        y_pred:
            Вектор прогнозов

        Returns:
        --------
        float:
            Значение mean absolute percentage error

        """
        try:
            res = np.mean(np.abs((y_true - y_pred) / y_true))
        except ZeroDivisionError:
            print("Вектор значений переменной содержит"
                  " нулевые значения")
        return res

    def calc_metrics(self, model, features, **kwargs) -> pd.DataFrame:
        """
        Функция расчета метрик качества модели регресии :
            MAE, MAPE, RMSE, R^2, person correlation, spearman correlation

        Parameters:
        ----------
        model:
            Объект обученной scikit-learn like модели

        features: list
            список переменных, которые использует модель

        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.


        Returns:
        -------
        pd.DataFrame:
            Датафрейм с метриками качества модели. В каждой строке
            хранится результат расчета метрики из index для всех датасетов в
            kwargs.
        """
        mae = {}
        mape = {}
        rmse = {}
        r2 = {}
        pearson = {}
        spearman = {}

        for ds_name, (X, y) in kwargs.items():
            pred_df = create_pred_df((self.model, self.features_list),
                                    X, y)
            if self.target_transformer is not None:
                y_trans = self.target_transformer.inverse_transform(pred_df["y_true"])
                y_pred_trans = self.target_transformer.inverse_transform(pred_df["y_pred"])
            else:
                y_trans = pred_df["y_true"]
                y_pred_trans = pred_df["y_pred"]

            mae[ds_name] = mean_absolute_error(y_trans, y_pred_trans)
            mape[ds_name] = mean_absolute_percentage_error(y_trans, y_pred_trans)
            rmse[ds_name] = root_mean_squared_error(y_trans, y_pred_trans)
            r2[ds_name] = r2_score(y_trans, y_pred_trans)
            pearson[ds_name] = pearsonr(y_trans, y_pred_trans)[0]
            spearman[ds_name] = spearmanr(y_trans, y_pred_trans)[0]

        res = {"MAE": mae, "MAPE": mape, "RMSE": rmse, "R2": r2,
               "Pearson": pearson, "Spearman": spearman}
        return pd.DataFrame(res).T

    def calc_correlations(self, **kwargs) -> pd.DataFrame:
        """
        Функция расчета линейной корреляции между целевой переменной и
        результатами предсказаний решающих деревьев, построенных на одном
        факторе.


        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.

        Returns:
        -------
        pd.DataFrame:
            Таблица с коэфф. корреляции для каждого признака по каждой
            выборке из kwargs

        """
        correlations = {}
        for ds_name, (X, y) in kwargs.items():
            corr_dt = DecisionTreeFeatureImportance(0, None)
            corr_dt.fit(X[self.features_list], y)
            correlations[f"{ds_name}"] = corr_dt.scores
        return pd.DataFrame(correlations)

    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Расчет стабильности метрик mae, mape, rse, pearson, spearman
        на всех наборах из kwargs. Запись результата на лист excel

        Parameters:
        ----------
        model
            объект обученной scikit-learn like модели

        features_list: list
            список переменных, которые использует модель

        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculation regression metrics stability report...")
        # считать метрики по модели
        metrics = self.calc_metrics(self.model, self.features_list, **kwargs)

        # считать изменения метрик по наборам данных
        for ds_name in metrics.columns:
            if ds_name != "train":
                metrics[f"delta_train_vs_{ds_name}"] = (metrics[ds_name] -
                                                        metrics["train"]) / \
                                                       metrics["train"]
        metrics = metrics.reset_index()
        metrics = metrics.rename(columns={"index": "metric"})

        # формат
        float_number_low = "## ##0.0000"
        float_percentage_low = "0.0000%"

        # записать результаты в excel-файл
        formats = {"num_format":
                       {float_percentage_low: ["delta_train_vs_test",
                                          "delta_train_vs_valid",
                                          "delta_train_vs_OOT"],
                        float_number_low: ["train", "test", "valid", "OOT"]}
                   }
        self._model_to_excel(df=metrics,
                             sheet_name="Regression Metrics",
                             formats=formats)

        # важность на основе корреляции по переменным
        corr_df = self.calc_correlations(**kwargs)

        # посчитать изменения коэф. корреляции
        for ds_name in corr_df.columns:
            if ds_name != "train":
                corr_df[f"delta_train_vs_{ds_name}"] = (corr_df[ds_name] -
                                                        corr_df["train"]) / \
                                                       corr_df["train"]

        corr_df.reset_index(inplace=True)
        corr_df.rename(columns={"index": "feature"}, inplace=True)

        # формат
        float_number_low = "## ##0.0000"
        float_percentage_low = "0.0000%"

        # записать результаты в excel-файл
        formats = {"num_format":
                       {float_percentage_low: ["delta_train_vs_test",
                                               "delta_train_vs_valid",
                                               "delta_train_vs_OOT"],
                    float_number_low: ["train", "test", "valid", "OOT"]}
                 }
        self._corr_to_excel(df=corr_df,
                       sheet_name="Regression Metrics",
                       pos=(metrics.shape[0]+2, 0),
                       formats=formats)
