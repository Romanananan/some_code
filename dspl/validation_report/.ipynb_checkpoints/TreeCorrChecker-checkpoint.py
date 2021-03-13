import pandas as pd
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from ..feature_extraction.transformers import DecisionTreeFeatureImportance
from ..utils import calculate_time_execute


class TreeCorrChecker(Checker):
    """

    """
    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 cat_features: list = None,
                 drop_features: list = None,
                 handbook=None):

        self.calc_dataset = "train"
        self.model_name = model_name
        self.writer = writer
        self.model = model
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.handbook = handbook

    def _to_excel(self, df: pd.DataFrame, sheet_name: str,
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
        excelWriter.write_data_frame(df, (0, 0), sheet_name, formats=formats)

        # Добавить описание
        worksheet = self.writer.sheets[sheet_name]
        worksheet.write_string(
            "E3", "Model flag - флаг,"
                  " означающий использование признака в модели")
        worksheet.write_string(
            "E2", "Correlation train - "
                  " корреляция между значением целевой переменной"
                  " и прогнозом дерева решений, построенном на одном признаке")

    def calc_correlations(self, x: pd.DataFrame, y:pd.Series) -> pd.DataFrame:
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
        corr_dt = DecisionTreeFeatureImportance(0, None)
        corr_dt.fit(x, y)
        correlations["Correlation Train"] = corr_dt.scores
        return pd.DataFrame(correlations)

    @calculate_time_execute
    # @calc_time(name="Decision tree correlation report")
    def validate(self, **kwargs):
        """
        Расчет коэфф. корреляии между целевой переменной и результатами
        предсказаний решающих древьев, построеными на каждой отдельно взятой
        фиче. Расчет изменения корр. коэфф. между выборками
        Запись результатов на лист excel

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Creating Decision tree correlation report...")
        x_train, y_train = kwargs.get("train", None)
        corr_df = self.calc_correlations(x_train.drop(self.cat_features,
                                                      axis=1),
                                        y_train)

        corr_df.reset_index(inplace=True)
        corr_df.rename(columns={"index": "feature"}, inplace=True)
        corr_df = corr_df.sort_values(by="Correlation Train", ascending=False)

        corr_df = corr_df.append(pd.DataFrame({"feature": self.cat_features,
                                     "Correlation Train":
                                         "Категориалльный признак"}),
                                 ignore_index=True)

        # Простановка флага использования в модели
        corr_df["Model flag"] = corr_df["feature"].isin(
            self.features_list).astype(int)

        # формат
        float_number_low = "## ##0.0000"

        # записать результаты в excel-файл
        formats = {"num_format": {float_number_low: ["Correlation Train"]}}
        self._to_excel(df=corr_df,
                       sheet_name="Correlation Coefficient",
                       formats=formats)
