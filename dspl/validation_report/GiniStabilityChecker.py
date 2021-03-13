import pandas as pd
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df
from ..utils import calculate_time_execute
from ..feature_extraction.transformers import GiniFeatureImportance


# Gini Variables stability Class
class GiniStabilityChecker(Checker):
    """
    Класс реализации проверки стабильности ранжирующей способности переменных,
    вошедших в модель на основе метрики Gini. Сравнение осуществляется на
    наборах train-valid, train-test, train-OOT

    Parameters:
    ----------
    writer: pd.ExcelWriter
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

    def __init__(self
                 , writer: pd.ExcelWriter
                 , model_name: str
                 , model
                 , features_list: list
                 , cat_features: list
                 , drop_features: list
                 , n_bins: int = 20
                 , current_path=None):
        
        self.writer = writer
        self.model_name = model_name
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model = model
        self.n_bins = n_bins
        self.current_path = current_path

    def _to_excel(self, df: pd.DataFrame, sheet_name: str, plot=False) -> None:
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

        float_number_low = "## ##0.0000"
        float_percentage_high = "0.00%"

        # Кастомный формат для таблицы
        fmt = {"num_format": {
            float_number_low: ["Gini train", "Gini test",
                                   "Gini valid", "Gini OOT", "Gini OOT_psi"]
            , float_percentage_high: ["delta_train_vs_valid",
                                      "delta_train_vs_OOT",
                                      "delta_train_vs_test",
                                      "delta_train_vs_OOT_psi"]}
        }

        bold_row = {"bold": {
            True: df.index[df["feature"] == "model"]}
        }
        
        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df.fillna("NA"), (0, 0), sheet=sheet_name,
                                     formats=fmt, row_formats=bold_row)

        # apply conditional format to highlight validation_report test results

        # gini delta on features FYI test
        for col in ["delta_train_vs_valid", "delta_train_vs_test",
                    "delta_train_vs_test2"]:
            if col in df.columns:
                excelWriter.set_simple_cond_format(df, (1, 0), col,
                                                   boundary=-0.25,
                                                   order="straight")
        for col in ["delta_train_vs_OOT", "delta_train_vs_OOT_psi"]:
            if col in df.columns:
                excelWriter.set_simple_cond_format(df, (1, 0), col,
                                                   boundary=-0.3,
                                                   order="straight")

        # gini delta on model
        row_num = df[df["feature"] == "model"].index[0]
        for col in ["delta_train_vs_valid", "delta_train_vs_test",
                    "delta_train_vs_test2"]:
            if col in df.columns:
                excelWriter.set_cell_cond_format(df, (0, 0), col, row_num,
                                                 upper=-0.25, lower=-0.15,
                                                 order="reverse")

        for col in ["delta_train_vs_OOT", "delta_train_vs_OOT_psi"]:
            if col in df.columns:
                excelWriter.set_cell_cond_format(df, (0, 0), col, row_num,
                                                 upper=-0.3, lower=-0.2,
                                                 order="reverse")

        if plot:
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                "A23", f"{self.current_path}/images/{sheet_name}.png")

        # Доп инфо
        sheet = self.writer.sheets[sheet_name]
        sheet.write_string(f"B{df.shape[0]+3}",
                           f"* - Значения метрики качества в разрезе "
                           f"факторов приведены для информации.{chr(10)}"
                           "Для валидации модели критична "
                           "стабильность метрики Gini по модели ")

    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Расчет показателей стабильности gini по переменным,
        вошедшим в модель, на указанных датасетах

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
        # ##
        # исключить из проверки категориальные фичи
        print("Calculating Gini stability report...")

        if self.cat_features is not None:
            gini_features_list = list(set(self.features_list)\
                                      - set(self.cat_features))
        else:
            gini_features_list = self.features_list

        transformer = GiniFeatureImportance(threshold=0,
                                            cat_features=None)

        for ds_name, (X, y) in kwargs.items():
            if ds_name == "train":

                features_df = X[gini_features_list]
                pred_df = create_pred_df(model_info=(self.model,
                                                      self.features_list),
                                          X=X[self.features_list], y=y)
                # переменные + прогноз
                features_df["model"] = pred_df["y_pred"]

                gini_imp = transformer.fit_transform(features_df, y)\
                    .drop("Selected", axis=1)
                gini_imp.columns = ["feature", f"Gini {ds_name}"]
                gini_imp.set_index("feature", inplace=True)

            else:
                features_df = X[gini_features_list]
                pred_df = create_pred_df(model_info=(self.model,
                                                     self.features_list),
                                         X=X[self.features_list], y=y)
                # переменные + прогноз
                features_df["model"] = pred_df["y_pred"]

                gini_imp_diff = transformer.fit_transform(features_df, y) \
                    .drop("Selected", axis=1)
                gini_imp_diff.columns = ["feature", f"Gini {ds_name}"]
                gini_imp_diff.set_index("feature", inplace=True)

                gini_imp = pd.concat([gini_imp, gini_imp_diff], axis=1)
                gini_imp[f"delta_train_vs_{ds_name}"] =\
                    (gini_imp[f"Gini {ds_name}"] - gini_imp[f"Gini train"]) \
                    / gini_imp["Gini train"].map(
                    lambda x: 0.00001 if x == 0. else x)

        gini_imp = gini_imp.reset_index()
        gini_imp = gini_imp.rename(columns={"index": "feature"})
        self._to_excel(gini_imp, "Variables GINI stability")
