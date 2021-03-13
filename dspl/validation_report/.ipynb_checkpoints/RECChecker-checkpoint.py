import pandas as pd
from sklearn.metrics import auc
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df, plot_rec_curve
from ..utils import calculate_time_execute


# REC analysis
class RECChecker(Checker):
    """
    Класс реализации анализа REC-кривой модели. Строит REC кривые для
    модели и некоторого baseline (средний прогноз). Считает отношение
    AOC_model / AOC_baseline.
    Только для задачи регресии
    Проверка осуществляется только на train наборе данных

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

    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 cat_features: list,
                 drop_features: list,
                 target_transformation=None,
                 plot_size=(10, 10),
                 current_path=None):

        self.writer = writer
        self.model_name = model_name
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model = model
        self.target_transformation = target_transformation
        self.plot_size = plot_size
        self.current_path = current_path

    def _to_excel(self, df: pd.DataFrame, sheet_name: str, plot=False,
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
        sheet = self.writer.sheets[sheet_name]

        # apply conditional format to highlight validation_report test results
        if "AOC (model/baseline)" in df.columns:
            excelWriter.set_col_cond_format_tail(df, (0, 0),
                                                 "AOC (model/baseline)",
                                                 upper=1, lower=1,
                                                 order="straight")

        if plot:
            # insert rec plot
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                f"A{df.shape[0] + 4}", f"{self.current_path}/images"
                                       f"/{sheet_name}.png")

        # Описание теста
        sheet.write_string(
            "L10", "AOC - площадь над кривой")

        sheet.write_string(
            "L12", "Accuracy, % - кумулятивная доля наблюдений в выборке")
        sheet.write_string(
            "L13", "absolute error - абсолютное значение ошибки прогноза")

        sheet.write_string(
            "L15", "baseline - модель, прогнозирующая среднее значение целевой"
                   " переменной для каждого наблюдения ")

        sheet.write_string(
            "L17", "Тест сравнивает построенную модель с baseline,")
        sheet.write_string(
            "L19", " если AOC_model > AOC_baseline,"
                   " то baseline оказался лучше")


    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Запуск процедуры анализа REC кривых.
        Записывает результаты на лист excel отчета "REC analysis"
        Сохраняет файл с кривыми
        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculating REC test...")
        x_train, y_train = kwargs.get("train", None)

        if x_train is None:
            raise KeyError("There must be train dataset included")

        pred_df = create_pred_df((self.model, self.features_list),
                                      x_train[self.features_list],
                                      y_train)

        if self.target_transformation is not None:
            y_trans = self.target_transformation.inverse_transform(
                pred_df["y_true"])
            y_pred_trans = self.target_transformation.inverse_transform(
                pred_df["y_pred"])
        else:
            y_trans = pred_df["y_true"]
            y_pred_trans = pred_df["y_pred"]

        # абсолютная ошибка на каждом наблюдении
        abs_error = abs(y_trans - y_pred_trans).sort_values()

        # baseline - среднее значение целевой переменной
        y_baseline_pred = y_trans.mean()
        abs_error_baseline = abs(y_trans - y_baseline_pred).sort_values()

        # кумулятивная ошибка на алгоритме
        len_error = len(abs_error)
        cum_accur = [(num+1)/len_error for num, _ in enumerate(abs_error)]

        # кумулятивная ошибка на бейзлайне
        len_error_baseline = len(abs_error_baseline)
        cum_accur_baseline = [(num+1)/len_error_baseline
                              for num, _ in enumerate(abs_error_baseline)]
        # отрисовка графика
        plot_rec_curve(abs_error, cum_accur,
                       abs_error_baseline, cum_accur_baseline,
                       name=f"{self.current_path}/images/REC curve")

        # расчет метрики aoc
        aoc_model = max(abs_error)*max(cum_accur) - auc(abs_error, cum_accur)
        aoc_baseline = max(abs_error_baseline)*max(cum_accur_baseline) \
            - auc(abs_error_baseline, cum_accur_baseline)

        rec = pd.DataFrame({
            "data set": "train",
            "AOC model": [aoc_model],
            "AOC baseline": [aoc_baseline],
            "AOC (model/baseline)": [aoc_model/aoc_baseline]
        })

        # формат
        float_number_low = "## ##0.0000"

        # записать результаты в excel-файл
        formats = {"num_format": {
                        float_number_low: ["AOC model", "AOC baseline",
                                           "AOC (model/baseline)"]}
                   }

        self._to_excel(rec, "REC curve", plot=True, formats=formats)
