import pandas as pd
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import plot_shap_importance
from ..utils import calculate_time_execute


# shap importance class
class ShapFeatureImportanceChecker(Checker):
    """
    Класс реализации проверки степени влияния на прогноза значений признака по
    методу SHAP.

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
                 , plot_size=(10, 10)
                 , current_path=None):

        self.writer = writer
        self.model_name = model_name
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model = model
        self.plot_size = plot_size
        self.current_path = current_path

        # lightGBM model is affected by deepcopying
        # save params for booster as shap requires
        if str(type(self.model)).endswith("lightgbm.sklearn.LGBMClassifier'>"):
            self.model.booster_.params = self.model.get_params()

    def _to_excel(self, sheet_name: str, plot=True) -> None:
        """
        Функция записи результатов проверка в excel файл на указанный
        лист и позицию

        Parameters:
        ----------
        sheet_name: str
            Имя листа, на который осуществить запись
        plot: bool
            Флаг необходимости добавить на страницу с отчетом график из файла
        """

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(pd.DataFrame(),
                                     (0, 0),
                                     sheet=sheet_name,
                                     formats={},
                                     row_formats=None)

        if plot:
            # Permutation importance plot
            sheet = self.writer.sheets[sheet_name]
            # shap summary
            sheet.insert_image(
                "P11", f"{self.current_path}/images/{sheet_name}.png")
            # shap impact dotted
            sheet.insert_image(
                "A11", f"{self.current_path}/images/{sheet_name}_summary.png")

        # Описание теста
        sheet.write_string(
            "E2", "Shap feature impact - метрика степени влияния"
                  " значений признака на прогноз модели")
        sheet.write_string(
            "E3", "подробное описание методики расчета: "
            "https://christophm.github.io/interpretable-ml-book/shap.html")
        sheet.write_string(
            "E4", "* - данный тест информативный")

    def _shap_calc(self, **kwargs):
        """
        Функция расчета SHAP feature importance.
        Реализует расчет shap values и сохранение следующих графиков:
            - Shap feature impact
            - Shap feature importance (mean)

        Вызывает метод сохранения графиков в excel-книгу

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        # shap importance summary
        sheet_name = "shap_feature_importance"
        plot_name = f"{self.current_path}/images/{sheet_name}_summary"
        plot_shap_importance_summary(estimator=self.model,
                                     df=kwargs["train"][0],
                                     features_list=self.features_list,
                                     name=plot_name)

        # shap importance dots
        plot_name = f"{self.current_path}/images/{sheet_name}"
        plot_shap_importance(estimator=self.model,
                             df=kwargs["train"][0],
                             features_list=self.features_list,
                             name=plot_name)

        self._to_excel(sheet_name, plot=True)

    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Инициирование запуска проверки SHAP feature importance.
        Проверка типа модели: для CatBoost нет корректной реализации данного
        метода

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculating shap features impact...")
        # Check for Catboost model input
        if str(type(self.model)).endswith("CatBoostClassifier'>"):
            print("Shap explainer does not support Catboost models yet")

        else:
            self._shap_calc(**kwargs)
