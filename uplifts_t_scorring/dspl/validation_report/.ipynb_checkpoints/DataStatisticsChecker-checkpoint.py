import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score
from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df
from ..utils import calculate_time_execute
from ..metrics import gini_score


# Calculate data statistics 
class DataStatisticsChecker(Checker):
    """
    Формирование описательной статистики наборов данных

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
    """
    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 cat_features: list = None,
                 drop_features: list = None,
                 model_type: str = "binary_classification",
                 target_transformer=None):
        
        self.model_name = model_name
        self.writer = writer
        self.model = model
        self.features_list = features_list
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model_type = model_type
        self.target_transformer = target_transformer

        self.formatWriter = FormatExcelWriter(writer=self.writer)


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
        if self.model_type == "binary_classification":
            data_dict = {}
            for data_name, (x, y) in kwargs.items():
                data_dict[data_name] = [
                    x.shape[0],
                    np.sum(y),
                    np.mean(y)
                ]
            data_stats = pd.DataFrame(data_dict).T
            data_stats = data_stats.reset_index()
            data_stats.columns = [
                "выборка",
                "# наблюдений",
                "# events",
                "# eventrate"
            ]
        elif self.model_type == "regression":
            data_dict = {}
            for data_name, (x, y) in kwargs.items():
                if self.target_transformer is not None:
                    y_transformed = self.target_transformer.inverse_transform(y)
                    data_dict[data_name] = y_transformed.describe()
                else:
                    data_dict[data_name] = y.describe()

            data_stats = pd.DataFrame(data_dict).T
            data_stats = data_stats.reset_index()
            data_stats.columns = ["выборка",
                                  "# наблюдений",
                                  "Target AVG-value",
                                  "Target STD-value",
                                  "Target MIN-value",
                                  "Target 25% percentile-value",
                                  "Target 50% percentile-value",
                                  "Target 75% percentile-value",
                                  "Target MAX-value"
                                  ]

        else:
            raise ValueError("model_type must be selected "
                             "as 'binary_classification' or 'regression'")

        return data_stats
        
        
    def _create_variables_stats(self, X, y):
        """
        Построение отчета о типах переменных из датасета.

        Parameters:
        -----------
        X: pd.DataFrame
            Датафрейм со значениями признаков

        y: pd.Series
            вектор со значениями целевой переменной

        Return:
        -------
        pd.DataFrame
            Описательная статистика исходного датасета:
                - Имя целевой переменной
                - количество категориальных признаков
                - Количество интервальных признаков
        """
        cat_features_num = 0 if self.cat_features is None else len(self.cat_features)
        data = pd.DataFrame({
            "Целевая переменная": [y.name],
            "# категории": cat_features_num,
            "# интервальные": X.shape[1] - cat_features_num
        })

        return data
        
    def _to_excel(self,
                  df: pd.DataFrame,
                  sheet: str,
                  pos: tuple,
                  width: int = None,
                  height: int = None,
                  formats: dict = None,
                  rows_format: dict = None):
        """
        Функция записи датафрейма в excel файл на указанный лист и позицию

        Parameters:
        ----------
        df: pd.DataFrame
            Датафрейм для записи в файл
        sheet: str
            Имя листа, на который осуществить запись
        pos:tuple
            x,y координаты левого верхнего угла датасета
        formats:dict
            Словарь с перечислением форматов для столбцов вида:
            {<формат> : [список столбцов для применения]},
            где формат задается стадартными excel-строками,
            например "## ##0" - целые числа с разделителем разрядов - пробел
        """
        formatWriter = FormatExcelWriter(writer=self.writer)
        formatWriter.write_data_frame(df=df,
                                      pos=pos,
                                      sheet=sheet,
                                      formats=formats,
                                      row_formats=rows_format)
        if height is not None:
            formatWriter.set_height(df, height, pos)
        if width is not None:
            formatWriter.set_width(df, width, pos)
        print(df)
        formatWriter.merge_cells(df=df, pos=pos, col_start="train",
                                 col_end="test", row_start=1, row_end=1)

    @staticmethod
    def calc_stats(df: pd.DataFrame) -> pd.DataFrame:
        """
        Функция расчета описательных статистик в разрезе каждой переменной

        Parameters:
        -----------
        df: pd.DataFrame
            Датафрейм для которого будут считаться статистики

        Returns:
        --------
        pd.DataFrame:
            Датарейм с описательными статистиками по каждой переменной:
                - кол-во зааполненных значений
                - доля заполненных значений
                - кол-во уникальных значений
                - среднее значение (только для числовых)
                - стандартное отклонение (только для числовых)
                - минимальное значение (только для числовых)
                - 25й перцениль (только для числовых)
                - 50й перцентиль (только для числовых)
                - 75й перцентель (только для числовых)
                - максимальное значени (только для числовых)

        """
        stats = df.describe(include="all").T.reset_index()

        # доля заполненных значений
        row_num = df.shape[0]
        fill_share = stats["count"] / row_num
        stats.insert(loc=2, column="share filled", value=fill_share)

        # кол-во уникальных значений
        unq = [ len(df[col].drop_duplicates().values) for col in df]
        stats.insert(loc=3, column="count unique", value=unq)
        
        # колонки для вывода
        stats = stats[["index", "count", "share filled", "count unique",
                       "mean", "std", "min", "25%", "50%", "75%", "max"]]
        stats.columns = [
            "Variable name",
            "# of filled values",
            "% of filled values",
            "# of unique values",
            "Variable AVG-value",
            "Variable STD-value",
            "Variable MIN-value",
            "Variable 25% percentile-value",
            "Variable 50% percentile-value",
            "Variable 75% percentile-value",
            "Variable MAX-value"
        ]

        return stats

    def _define_model_type(self) -> str:
        """
        Определяет какая модель лежит в self.model
        Возвращает строковый признак
        """
        return str(type(self.model)).replace("'>", "").split(".")[-1]

    def print_title_page(self, **kwargs):

        # model title
        title_df = pd.DataFrame()
        title_df["Parameter"] = ["Model name",
                                 "Model description",
                                 "Observation dates",
                                 "Population requirements",
                                 "Outcome period",
                                 "Outcome variables",
                                 "Used algorithm"]
        for ds in kwargs.keys():
            title_df[ds] = \
                            [f"<Заполните название модели> {chr(10)}например:"
                             f" Модель прогнозирования оттока клиентов по"
                             f" пакету услуг Премьер",

                             f"<Заполните описание модели: цель моделирования,"
                             f"  для какого бизнес процесса строится модель, "
                             f"какие данные используются >  {chr(10)}  "
                             f"например: Модель для выделения клиентов банка "
                             f"наиболее склонных к закрытию Пакета Услуг "
                             f"Премьер. {chr(10)} Модель строится на данных "
                             f"ЦОД по информации об открытии/закрытии пакетов "
                             f"услуг клиентами банка. {chr(10)} В качестве "
                             f"факторов используются данные клиентского "
                             f"профиля трайба Массовая Персонализация.",

                             f"<Укажите отчетные даты> {chr(10)} например: "
                             f"31.01.2019, 28.02.2019, 31.03.2019",


                             f"<Заполните критерии отбора популяции: "
                             f"фильтры, исключения>{chr(10)}например:{chr(10)}"
                             f"1. Клиенты, по которым есть информация в "
                             f"витрине клиентского профиля на отчетную дату."
                             f"{chr(10)}2. Бизнес-клиенты Банка (валидные ФИО," 
                             f" ДУЛ).{chr(10)} 3. Клиенты, по которым открыт "
                             f"Пакет Услуг Премьер на отчетную дату.",

                             f"<Укажите период, за который расчитывалась "
                             f"целевая переменная>{chr(10)} например: 3 месяца"
                             f" от (даты наблюдения + 1 месяц)",

                             "<Заполните определние целевого события/"
                             "переменной>",

                             f"{self._define_model_type()}"]

        # model params
        dataset_df = pd.DataFrame(columns=kwargs.keys())

        for ds_name, (x, y) in kwargs.items():
            pred_df = create_pred_df((self.model, self.features_list), x, y)
            if self.model_type =="regression":
                dataset_df[ds_name] = [len(y), y.mean(),
                                       mean_absolute_error(pred_df["y_true"],
                                                           pred_df["y_pred"])]
                dataset_df.index = ["# observations", "AVG Target value",
                                    "MAE"]
            elif self.model_type == "binary_classification":
                dataset_df[ds_name] = [len(y), y.mean(),
                                       gini_score(pred_df["y_true"],
                                                  pred_df["y_pred"])]
                dataset_df.index = ["# observations", "Event Rate",
                                    "Gini"]
            else:
                raise KeyError(f"Model type {self.model_type} is "
                               f"not supported")

        dataset_df.reset_index(inplace=True)
        dataset_df.rename(columns={"index": "Parameter"}, inplace=True)

        dataset_format = {"bold": {True: ["Parameter"]},
                          "align": {"center": ["Parameter", "train", "valid",
                                               "OOT", "test", "test2"]},
        }

        if self.model_type == "binary_classification":
            rows_ds_format = {
                "num_format": {"## ##0": [0],
                               "0.00%": [1, 2]}
            }
        else:
            rows_ds_format = {
                "num_format": {"## ##0": [0],
                               "## ##0.00": [1, 2]}
            }

        self.formatWriter.write_data_frame(dataset_df,(title_df.shape[0]+2, 0),
                                           "Model Summary", dataset_format,
                                           rows_ds_format)

        #self._to_excel(dataset_df, "Model Summary", (title_df.shape[0]+2, 0),
        #               formats=dataset_format, rows_format=rows_ds_format)

        title_format = {"bold": {True: ["Parameter"]},
                        "align": {"center": ["Parameter", "train", "valid",
                                             "test", "test2", "OOT","OOT_psi"]},
                        "valign": {"vcenter": ["Parameter", "train", "valid",
                                               "test", "test2", "OOT","OOT_psi"]},
                        "font_color": {"#BFBFBF": ["train", "valid",
                                       "test", "test2", "OOT","OOT_psi"]},
                        "text_wrap": {True: ["train", "valid",
                                             "test", "test2", "OOT","OOT_psi"]}
        }
        self.formatWriter.write_data_frame(title_df, (0, 0),  "Model Summary",
                                           title_format)
        self.formatWriter.set_width(title_df, 30, (0, 0))
        self.formatWriter.set_height(title_df, 60, (0, 0))

        full_merge = ["Model name", "Model description", "Outcome variables",
                      "Used algorithm"]
        oot_merge = ["Observation dates", "Population requirements",
                     "Outcome period"]
        full_merge_end = list(kwargs.keys())[-1]

        if "OOT" in kwargs.keys():
            oot_merge_end = list(kwargs.keys())[-3]
        else:
            oot_merge_end = full_merge_end

        for param in full_merge:
            rows = title_df.index[title_df["Parameter"] == param]
            self.formatWriter.merge_cells(df=title_df, pos=(0, 0),
                                          col_start="train",
                                          col_end=full_merge_end,
                                          row_start=rows.min(),
                                          row_end=rows.max())
        for param in oot_merge:
            rows = title_df.index[title_df["Parameter"] == param]
            self.formatWriter.merge_cells(df=title_df, pos=(0, 0),
                                          col_start="train",
                                          col_end=oot_merge_end,
                                          row_start=rows.min(),
                                          row_end=rows.max())
        # merge OOT columns
        if "OOT" in kwargs.keys():
            for param in oot_merge:
                rows = title_df.index[title_df["Parameter"] == param]
                self.formatWriter.merge_cells(df=title_df, pos=(0, 0),
                                              col_start="OOT",
                                              col_end="OOT_psi",
                                              row_start=rows.min(),
                                              row_end=rows.max())

    def print_model_steps_page(self):

        # model title
        steps_df = pd.DataFrame()
        steps_df["Этап построения"] = ["Сбор данных",
                                       "Разбиение выборок",
                                       "Обработка данных",
                                       "Отбор признаков",
                                       "Построение моделей",
                                       "Подбор гиперпараметров модели "
                                       "(диапазон поиска )"]

        steps_df["Описание"] = ["<Прикрепите скрипт формирования выборки "
                                "для обучения>",
                                "<Заполните описание алгоритма разбиения "
                                "выборки на train, test, valid, OOT>",
                                "< Заполните использованные методы "
                                "предобработки признаков: заполнение пропусков,"
                                " кодирование категорий и др.>",
                                "< Заполните список алгоритмов, использованных"
                                "для отбора признаков>",
                                "< Зполните использованные методы построения "
                                "моделей>",
                                "< Заполните области поиска гиперпараметров>"
                            ]

        steps_df_format = {"bold": {True: ["Этап построения"]},
                           "align": {"center": ["Этап построения", "Описание"]},
                           "valign": {"vcenter": ["Этап построения",
                                                  "Описание"]},
                           "font_color": {"#BFBFBF": ["Описание"]}
        }
        self.formatWriter.write_data_frame(steps_df, (0, 0), "Modelling Steps",
                                           steps_df_format)
        self.formatWriter.set_height(steps_df, 60, (0, 0))

    # @calc_time(name="Data statistics report ")
    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Функция вызова методов класса для создания каждой части отчета:
            - статистики в разрезе переменных
            - характеристики исходных наборов данных ( размерность, eventrate)
            - количество интервальных и категоральных переменых

        , а также вызовы методов для записи каждой части отчета
        с указаными форматами столбцов

        Parameters:
        ----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.

        """
        print("Creating data statistics report...")

        # Model summary page
        self.print_title_page(**kwargs)
        self.print_model_steps_page()
        # Data statistics page
        sample_stats = self.create_data_stats(**kwargs)
        sample_key = next(iter(kwargs))
        variable_stats = self._create_variables_stats(*kwargs[sample_key])

        sample_describe = self.calc_stats(kwargs[sample_key][0])

        # стандартные форматы чисел
        int_number = "## ##0"
        float_percentage = "0.00%"
        float_number_high = "## ##0.00"
        
        formats = {"num_format": {int_number: ["# наблюдений", "# events"],
                                  float_percentage: ["# eventrate"]}
                   }

        self.formatWriter.write_data_frame(df=sample_stats,
                                           sheet="Data Statistics",
                                           pos=(0, 0), formats=formats)

        formats = {"num_format": {int_number: ["# категории",
                                               "# интервальные"]}
                   }
        self.formatWriter.write_data_frame(df=variable_stats,
                                           sheet="Data Statistics",
                                           pos=(sample_stats.shape[1] + 1, 0),
                                           formats=formats)

        formats = {"num_format": {int_number: ["# of filled values",
                                               "# of unique values"],
                                  float_percentage: ["% of filled values"],
                                  float_number_high:
                                      ["Variable AVG-value",
                                       "Variable STD-value",
                                       "Variable MIN-value",
                                       "Variable 25% percentile-value",
                                       "Variable 50% percentile-value",
                                       "Variable 75% percentile-value",
                                       "Variable MAX-value"]}
                   }
        self.formatWriter.write_data_frame(df=sample_describe,
                                           sheet="Data Statistics",
                                           pos=(sample_stats.shape[1] + variable_stats.shape[1] + 2, 0),
                                           formats=formats)