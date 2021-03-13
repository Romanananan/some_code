import pandas as pd
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df
from ..utils import calculate_time_execute
from ..reports.report import Regression_DM, Binary_DM
from ..plots import plot_regression_graph, plot_binary_graph


# Prediction bins statistics checker
class BinsMetricsChecker(Checker):
    """
    Реализация расчета статистик по бинами прогноза модели и по модели в целом.
    Расчет производится для каждого набора данных.

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
    
    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 n_bins: int = 20,
                 cat_features: list = None,
                 model_type: str = "binary_classification",
                 target_transformer=None,
                 current_path: str = None):
        
        self.calc_dataset = "train"
        self.writer = writer
        self.model_name = model_name
        self.model = model
        self.features_list = features_list
        self.cat_features = cat_features
        self.model_type = model_type
        self.target_transformer = target_transformer
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

        int_number = "## ##0"
        float_number_high = "## ##0.00"

        # Кастомный формат для таблицы
        fmt = { "num_format":
                     {int_number: ["bin", "#obs"],
                      float_number_high: ["pred_min", "pred_mean", "pred_max",
                                "real_min", "real_mean", "real_max",
                                "real 25p", "real 50p", "real 75p", "MAE",
                                "MAPE", "RMSE", "R2", "pearson-correlation",
                                "spearman-correlation"]
                      }
                }

        bold_row = {"bold": {
            True: df.index[df["bin"] == "Total"]}
        }

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df.fillna("NA"), (0, 0), sheet=sheet_name,
                                     formats=fmt, row_formats=bold_row)

        if plot:
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                "A23", f"{self.current_path}/images/{sheet_name}.png")

    def _get_estimator_params(self) -> pd.DataFrame():
        """
        Достает список и значения параметров из модели
        Сохраняет в pandas dataframe
        """

        try:
            params = self.model.params.copy()
        except AttributeError:
            params = self.model.get_params()

        if hasattr(self.model, "n_iterations"):
            n_iters = self.model.n_iterations or params["n_estimators"]
            if isinstance(n_iters, int):
                params["n_estimators"] = n_iters

        params_df = pd.io.json.json_normalize(params).T.reset_index()
        params_df.columns = ["Parameter", "Value"]
        
        return params_df

    def create_classification_report(self, **kwargs):
        """
        Рассчет метрик и построение графиков по моделям бинарной
        классификации.

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """

        transformer = Binary_DM(n_bins=20)
        for data_name, (x, y) in kwargs.items():

            sheet_name = f"{data_name}_{self.model_name}"
            model_tuple = (self.model, self.features_list)
            pred_df = create_pred_df(model_tuple, x, y)
            y_true = pred_df["y_true"]
            y_pred = pred_df["y_pred"]
            data = transformer.transform(y_true, y_pred)

            # график
            plot_binary_graph(y_true, y_pred,
                                  f"{self.current_path}/images/{sheet_name}")
            # записать таблицу на лист
            self._to_excel(df=data, sheet_name=sheet_name, plot=True)

    def create_regression_report(self, **kwargs):
        """
        Рассчет метрик и построение графиков по моделям регресии.

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """

        transformer = Regression_DM(self.target_transformer, n_bins=21)
        for data_name, (x, y) in kwargs.items():

            sheet_name = f"{data_name}_{self.model_name}"
            model_tuple = (self.model, self.features_list)
            pred_df = create_pred_df(model_tuple, x, y)

            if self.target_transformer is not None:
                y_true = self.target_transformer.\
                    inverse_transform(pred_df["y_true"])
                y_pred = self.target_transformer.\
                    inverse_transform(pred_df["y_pred"])
            else:
                y_true = pred_df["y_true"]
                y_pred = pred_df["y_pred"]

            data = transformer.transform(y_true, y_pred)

            # график
            plot_regression_graph(y_true, y_pred,
                                  f"{self.current_path}/images/{sheet_name}")
            # записать таблицу на лист
            self._to_excel(df=data, sheet_name=sheet_name, plot=True)



    @calculate_time_execute
    # @calc_time(name="Prediction bins statistics and quality metrics")
    def validate(self, **kwargs):
        """
        Функция запуска расчетов статистик по бинам для каждого набора данных
        из словаря kwargs
        Дополнительно реализовано извлечение параметров объекта модели
        и сохранений таблицей на отдельный лист

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.

        """
        print("Creating bins prediction report")

        # гиперпараметры модели
        model_params = self._get_estimator_params()
        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(model_params,
                                     (0, 0),
                                     sheet=f"{self.model_name}",
                                     formats=None)

        if self.model_type == "binary_classification":
            self.create_classification_report(**kwargs)

        elif self.model_type == "regression":
            self.create_regression_report(**kwargs)
