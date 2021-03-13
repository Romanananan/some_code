import pandas as pd
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from ..utils import calculate_time_execute
from ..feature_extraction.transformers import GiniFeatureImportance


class GiniVariablesChecker(Checker):
    """
    Реализация расчета метрики gini по каждой переменной набора train

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

    cat_features: list
        Список категориальных признаков

    drop_features: list
        Список мусорных признаков для исключения из анализа

    handbook: pd.DataFrame
        Датафрейм с ручным справочником для расшифровки смысла переменных.
        Должен из двух полей : variable, description
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

    def _to_excel(self,
                  df: pd.DataFrame,
                  sheet_name: str,
                  pos: tuple,
                  formats: dict,
                  plot: bool = False) -> None:
        """
        Функция записи датафрейма в excel файл на указанный лист и позицию

        Parameters:
        ----------
        df: pd.DataFrame
            Датафрейм для записи в файл
        sheet_name: str
            Имя листа, на который осуществить запись
        pos:tuple
            x,y координаты левого верхнего угла датасета
        formats:dict
            Словарь с перечислением форматов для столбцов вида:
            {<формат> : [список столбцов для применения]},
            где формат задается стадартными excel-строками,
            например "## ##0" - целые числа с разделителем разрядов - пробел
        plot: bool
            Флаг необходимости добавить на страницу с отчетом график из файла
        """

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df=df,
                                     pos=pos,
                                     sheet=sheet_name,
                                     formats=formats)

        if plot:
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                f"A{df.shape[0]}", f"{self.current_path}/images/{sheet_name}.png")

        # Добавить описание
        worksheet = self.writer.sheets[sheet_name]
        worksheet.write_string(
            "E2", "Model flag - флаг, означающий использование признака в модели")

    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Расчет gini importance для каждой переменной из train датафрейма.
        Для признаков типа object, category, а также из списка
        categorical_features расчет не выполняется

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculating variables gini report...")
        X, y = kwargs.get("train", None)

        if X is not None:
            categorical_features = []

            # Признаки, имеющие тип object и category считаются категориальными
            categorical_features += X.select_dtypes(include=["object",
                                                             "category"])\
                .columns.tolist()
            # Добавить к найденным категориальным переменным
            # список категориальных переменных из конфиг. файла
            if self.cat_features is not None:
                categorical_features += self.cat_features
            categorical_features = list(set(categorical_features))
                
            # Посчитать gini importance для НЕ категориальных
            transformer = GiniFeatureImportance(
                threshold=0,
                cat_features=categorical_features)
            gini_imp = transformer.fit_transform(X, y).drop("Selected", axis=1)

            gini_imp.columns = ["Variable", "Gini train"]

            # Простановка флага использования в модели
            gini_imp["Model flag"] = gini_imp["Variable"].isin(
                self.features_list).astype(int)

            # Описание переменных
            if self.handbook is not None:
                gini_imp = gini_imp.merge(self.handbook["description"],
                                          how="left",
                                          left_on="Variable",
                                          right_index=True)

            float_number_low = "## ##0.0000"

            fmt = {"num_format": {float_number_low: ["Gini train"]} }

            # Записать результат в файл
            self._to_excel(df=gini_imp,
                           sheet_name="Variables GINI",
                           pos=(0, 0),
                           formats=fmt,
                           plot=False)

        else:
            raise KeyError("train dataset must be passed to function")
