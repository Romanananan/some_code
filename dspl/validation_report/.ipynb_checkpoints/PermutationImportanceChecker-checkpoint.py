import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import plot_permutation_importance, \
                   calculate_permutation_feature_importance
from ..utils import calculate_time_execute


# Permutation importance class
class PermutationImportanceChecker(Checker):
    """
    Класс реализации проверки важности признаков на основе перестановок.
    Расчитывается относительное изменения метрики gini при случайном
    перемешивании значений признака. Проверка выполняется для каждого признака
    вошедшего в модель

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
                 model_type: str = "binary_classification",
                 plot_size=(10, 10),
                 current_path=None):

        self.writer = writer
        self.model_name = model_name
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model = model
        self.model_type = model_type
        if model_type == "binary_classification":
            self.metric = roc_auc_score
        elif model_type == "regression":
            self.metric = self._correlation_metric
        else:
            raise ValueError("model_type must be 'binary_classification' "
                             "or regression")
        self.plot_size = plot_size
        self.current_path = current_path

    def _correlation_metric(self, y_true, y_pred) -> float:
        return abs(spearmanr(y_true, y_pred)[0])

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
        
        # Кастомный формат для таблицы
        fmt = {"num_format":
                   {float_number_low: ["permutation_importance_test",
                                       "permutation_importance_valid",
                                       "permutation_importance_OOT"]}
              }

        bold_row = {"bold":{
            True: df.index[df['feature'] == "factors_relevancy"]}
        }

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df, (0, 0), sheet=sheet_name,
                                     formats=fmt, row_formats=bold_row)

        # apply conditional format to highlight validation_report test results
        for col in ["permutation_importance_test"
                    , "permutation_importance_valid"
                    , "permutation_importance_OOT"]:
            if col in df.columns:
                excelWriter.set_col_cond_format_tail(df,
                                                     (0, 0),
                                                     col,
                                                     lower=200,
                                                     upper=0.0,
                                                     order="reverse")

        if plot:
            # Permutation importance plot
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                f"A{df.shape[0] + 4}", f"{self.current_path}/images"
                                       f"/{sheet_name}.png")

        # Описание теста
        sheet.write_string(
            "E2", "Permutation importance - метрика важности "
                  "признака в построенной модели")
        sheet.write_string(
            "E3", f"Считается как относительное изменение метрики"
                  f" качества модели ({self.metric.__name__})"
                  f" при перемешивании значений признака")
        sheet.write_string(
            "E5", "Factors relevancy - доля факторов с "
                  "важностью 20% и более от фактора с максимальной важностью")
        sheet.write_string(
            "E7", "* - данный тест информативный")

    def _calc_perm_importance(self, **kwargs) -> pd.DataFrame:
        """
        Расчет важности признаков вошедших в модели на основе метода
        перестановок.

        Считает на датасетах :
            test/valid
            oot

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.

        Returns:
        -------
        pd.DataFrame
            датасет с полсчитанной важностью признаков вида:
                - feature_name
                - Permutation importance test/valid
                - Permutatuin importance oot
        """
        X_test, y_test = kwargs.get("test", (None, None))
        X_valid, y_valid = kwargs.get("valid", (None, None))
        X_test2, y_test2 = kwargs.get("test2", (None, None))
        X_OOT, y_OOT = kwargs.get("OOT", (None, None))

        perm_importance_final = pd.DataFrame()

        # Посчитать PI на test или valid
        if X_test is not None:
            perm_importance_final = calculate_permutation_feature_importance(
                                        self.model,
                                        self.metric,
                                        X=X_test[self.features_list],
                                        y=y_test,
                                        fraction_sample=0.95)

            perm_importance_final.rename(
                columns={"permutation_importance":
                         "permutation_importance_test"},
                inplace=True)
            perm_importance_final.set_index("feature", inplace= True)
                                                                                                  
        elif X_valid is not None:
            perm_importance_final = calculate_permutation_feature_importance(
                                        self.model,
                                        self.metric,
                                        X=X_valid[self.features_list],
                                        y=y_valid,
                                        fraction_sample=0.95)
                
            perm_importance_final.rename(
                columns={"permutation_importance":
                         "permutation_importance_valid"},
                inplace=True)
                                                      
            perm_importance_final.set_index("feature", inplace=True)

        # Посчитать PI на OOT при наличии
        if X_OOT is not None:
            perm_importance = calculate_permutation_feature_importance(
                self.model,
                self.metric,
                X=X_OOT[self.features_list],
                y=y_OOT,
                fraction_sample=0.95)

            perm_importance.rename(columns={"permutation_importance":
                                            "permutation_importance_OOT"},
                                   inplace=True)
                                                                           
            perm_importance.set_index("feature", inplace=True)
            perm_importance_final = pd.concat([perm_importance_final,
                                               perm_importance],
                                              axis=1)
        # Посчитать PI на test2 при наличии
        if X_test2 is not None:
            perm_importance = calculate_permutation_feature_importance(
                self.model,
                self.metric,
                X=X_test2[self.features_list],
                y=y_test2,
                fraction_sample=0.95)

            perm_importance.rename(columns={"permutation_importance":
                                                "permutation_importance_test2"},
                                   inplace=True)

            perm_importance.set_index("feature", inplace=True)
            perm_importance_final = pd.concat([perm_importance_final,
                                               perm_importance],
                                              axis=1)
        return perm_importance_final
        
    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Запуск процедуры расчета важности признаков.
        Добавляет в итоговоый датасет factors relevancy :
            доля признаков с важностью 20% и более от признака
            с максимальным значением важности
        Рисует и сохраняет график permutation importance plot.
        Записывает результат проврки на страницу excel отчета
        "Permutation importance"

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculating permutation importance...")

        PI = self._calc_perm_importance(**kwargs)

        psi = []
        cols = []
        
        for col in PI.columns:
            PI_share = 100*PI[col]/PI[col].max()
            psi.append(100 * ((PI_share > 20).sum())/PI_share.count() )
            cols.append(col)
            
        psi_df = pd.DataFrame(data=[psi]
                              , columns=cols
                              , index=["factors_relevancy"])
        
        # сохранить график :
        # сортирвока по относительному изменению
        PI = PI.sort_values(by=PI.columns[0], ascending=False)
        PI_plot = PI.reset_index()
        PI_plot.rename(columns={"index": "feature"}, inplace=True)

        # permutation importance
        sheet_name = "perm_importance"
        plot_name = f"{self.current_path}/images/{sheet_name}"
        plot_permutation_importance(PI_plot
                                    , x_column="feature"
                                    , name=plot_name)

        # добавить в таблицу строку с релевантностью факторов в %
        PI = PI.append(psi_df)
        PI.reset_index(inplace=True)
        PI.rename(columns={"index": "feature"}, inplace=True)
        
        self._to_excel(PI, sheet_name, plot=True)
