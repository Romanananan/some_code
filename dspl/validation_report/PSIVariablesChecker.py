import pandas as pd
import numpy as np

from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter
from .funcs import create_pred_df

from ..utils import calculate_time_execute


# PSI Calculation class
class PSIVariablesChecker(Checker):
    """
    Класс реализации проверки population stability index
    по переменным используемым в модели и бинам прогнозов.

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
                 features_list=list,
                 cat_features: list = None,
                 drop_features: list = None,
                 model_type: str = "binary_classification"):

        self.writer = writer
        self.model = model
        self.features_list = features_list
        self.model_name = model_name
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.model_type = model_type
        # Датафреймы для хранения результатов проверки 
        self.psi_short = pd.DataFrame()
        self.psi_detailed = pd.DataFrame()

    def _to_excel(self, df: pd.DataFrame, sheet_name: str, fmt=None) -> None:
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
        bold_row = {"bold": {
            True: df.index[df["feature"] == "y_pred"]}
        }

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df, (0, 0), sheet_name, fmt,
                                     row_formats=bold_row)

        # apply conditional format to highlight validation_report test results
        for col in ["PSI_train_vs_valid_events",
                    "PSI_train_vs_valid_all",
                    "PSI_train_vs_valid_nevents",
                    "PSI_train_vs_OOT_all",
                    "PSI_train_vs_OOT_events",
                    "PSI_train_vs_OOT_nevents",
                    "PSI_train_vs_test_all",
                    "PSI_train_vs_test_events",
                    "PSI_train_vs_test_nevents",
                    "PSI_train_vs_valid",
                    "PSI_train_vs_OOT",
                    "PSI_train_vs_test",
                    "PSI_train_vs_test2_all",
                    "PSI_train_vs_OOT_psi",
                    "PSI_train_vs_OOT_psi_all",
                    "PSI_train_vs_OOT_psi_events",
                    "PSI_train_vs_OOT_psi_nevents",
                    ]:
            if col in df.columns:
                excelWriter.set_simple_cond_format(df, (0, 0), col,
                                                   boundary=0.2,
                                                   order="reverse")

            # Доп инфо
            sheet = self.writer.sheets[sheet_name]
            sheet.write_string(f"B{df.shape[0] + 3}",
                               f" * - Значения Population stability index "
                               f"в разрезе факторов приведены для "
                               f"информации")

    def _create_checklist(self, df_list: list):
        """
        Проверка наличия датасетов в словаре и формирование списка для
        сравнений с train

        Parameters:
        -----------
        df_list: list
            Список датасетов

        Returns:
        list
            список датасетов по которым нужно произвести провверку
        -------
        """
        check_list = []

        # Создать список для проверки 
        if "test" in df_list:
            check_list.append("test")
        elif "valid" in df_list:
            check_list.append("valid")

        if "OOT" in df_list:
            check_list.append("OOT")

        if "OOT_psi" in df_list:
            check_list.append("OOT_psi")

        if "test2" in df_list:
            check_list.append("test2")

        return check_list

    def create_df(self, x: pd.DataFrame,
                  y: pd.Series, model, features: list) -> dict:
        """
        Функция создает на выходе 3 датафрема: полный, events, no events

        Parameters:
        -----------
        x:pd.DataFrame
            Датафрейм с признаками
        y:pd.Series
            Истинные значения целевой переменной
        model
            sklearn-like модель, после применения метода fit
        features:list
            список переменных используемых в модели

        Returns:
        --------
        dict
            словарь с разбитыми датасетами, ключи:
            "all", "events", "nevents"
        """
        y_pred = create_pred_df(model_info=(self.model,
                                            self.features_list),
                                X=x, y=y)["y_pred"]
        if self.model_type == "binary_classification":
            full_df = pd.concat([x[features], y_pred, y], axis=1)
            events_df = full_df[full_df[y.name] == 1]
            nevents_df = full_df[full_df[y.name] == 0]
            res = {"all": full_df,
                   "events": events_df,
                   "nevents": nevents_df}

        elif self.model_type == "regression":
            full_df = pd.concat([x[features], y_pred, y], axis=1)
            res = {"all": full_df}

        return res

    def cut_buckets_groups(self, df: pd.DataFrame, df_name: str, perc: dict)\
            -> pd.DataFrame:
        """
        Разбиение всех столбцов в pd.DataFrame по укзаанными в dict пороговым
        значениям. Группировка и подсчет доли наблюдений в каждом интервале.

        Parameters:
        -----------
        df: pd.DataFrame
            датафрейм с исходными значениями переменных

        df_name: str
            имя набора данных (train, valid, test, oot)

        perc: dict
            словарь с необходимыми пороговыми значениями для разбиения
            по каждоый фиче вида:
            {<название фичи>: [<список пороговых значений>]}

        Returns:
        --------
        pd.DataFrame
            Датафрейм с группировкой фичей по бакетами и долей-количеством
            наблюдений в каждом бакете
        """

        # выходной датафрейм
        out_stats = pd.DataFrame()
        for col in df.columns:
            # для хранения порогов разбиения на перцентили
            missings = df[df.isna()[col]][col]

            # выделить отдельно пропуски
            missing_cnt = np.nan if len(missings) == 0 else len(missings)
            missing_stats = pd.DataFrame({"feature": [col],
                                          "bucket": ["MISSING"],
                                          f"obs_count_{df_name}":[missing_cnt],
                                          })

            # разбить на бакеты остальные значения
            values = df[df[col].notna()][col]
            if col not in perc.keys():
                perc[col] = np.unique([np.percentile(values
                                                     , interpolation="lower"
                                                     , q=q) for q in
                                       np.arange(0, 101, 10)])
            buckets = pd.cut(x=values,
                             bins=perc[col],
                             duplicates="drop",
                             include_lowest=True,
                             labels=False).rename("bucket")

            # Склеить номера бакетов со значениями наблюдений
            buckets_group = pd.concat([values, buckets], axis=1)

            # min-max статистики только для train выборки
            if df_name == "train":
                buckets_group = buckets_group.groupby("bucket").agg(
                    ["min", "max", "count"])[col]\
                    .rename(columns={"min": "min_value",
                                     "max": "max_value",
                                     "count": f"obs_count_{df_name}"})
            else:
                buckets_group = buckets_group.groupby("bucket").agg(
                    ["count"])[col].rename(
                                    columns={"count": f"obs_count_{df_name}"})

            # buckets_group.columns = buckets_group.columns.droplevel()
            buckets_group = buckets_group.reset_index()\
                .rename(columns={"index": "bucket"})
            buckets_group["feature"] = col

            # Добавить пропуски по фиче
            if len(missings)>0:
                buckets_group = buckets_group.append(missing_stats,
                                                     ignore_index=True)

            # Доля наблюдений в каждой выборке
            buckets_group[f"obs_share_{df_name}"] = \
                buckets_group[f"obs_count_{df_name}"] \
                / buckets_group[f"obs_count_{df_name}"].sum()

            #  Добавить всю статистику по фиче в финальный датасет
            out_stats = out_stats.append(buckets_group, ignore_index=True)
        return out_stats

    def calc_buckets_categories(self, df: pd.DataFrame, df_name: str)\
            -> pd.DataFrame:
        """
        Группировка признаков по уникальным значениям
        и подсчет доли-количества наблюдений в каждом интервале.

        Parameters:
        -----------
        df: pd.DataFrame
            датафрейм с исходными значениями переменных

        df_name: str
            имя набора данных (train, valid, test, oot)

        Returns:
        --------
        pd.DataFrame
            Датафрейм с группировкой фичей по значениям и долей-количеством
            наблюдений в каждом бакете
        """

        out_stats = pd.DataFrame()

        for col in df.columns:
            missings = df[df.isna()[col]][col]

            # выделить отдельно пропуски
            missing_cnt = np.nan if len(missings) == 0 else len(missings)
            missing_stats = pd.DataFrame({"feature": [col],
                                          "bucket": ["MISSING"],
                                          f"obs_count_{df_name}": [missing_cnt]
                                          })
            # категории без пропусков 
            values = df[df.notna()[col]][col]

            if df_name == "train":
                groups = values.groupby(by=values.astype(str))\
                    .agg(["min", "max", "count"])\
                    .rename(columns={"min": "min_value",
                                     "max": "max_value",
                                     "count": f"obs_count_{df_name}"})
            else:
                groups = values.groupby(by=values.astype(str))\
                    .agg(["count"])\
                    .rename(columns={"count": f"obs_count_{df_name}"})

            groups = groups.reset_index().rename(columns={col: "bucket"})
            groups["feature"] = col

            # Добавить пропуски
            if len(missings) is not None:
                groups = groups.append(missing_stats, ignore_index=True)

            # Посчитать доли
            groups[f"obs_share_{df_name}"] = groups[f"obs_count_{df_name}"] \
                                             / groups[f"obs_count_{df_name}"]\
                                             .sum()

            out_stats = out_stats.append(groups, ignore_index=True)
        return out_stats

    def calc_psi_pair(self,
                      base_df: pd.DataFrame,
                      base_df_name: str,
                      diff_df: pd.DataFrame,
                      diff_df_name: str,
                      features_type: str = "numeric") -> pd.DataFrame:
        """
        Расчет PSI по паре датафреймов: base_df vs diff_df для конкретных
        наборов признаков "numeric" или "categorical"

        Parameters:
        -----------
        base_df: pd.DataFrame
            Датафрейм относительно которого будет считаться PSI

        base_df_name: str
            Имя основного датасета

        diff_df: pd.DataFrame
            Датафрейм на котором будет считаться PSI

        diff_df_name: str
            Имя датасета для расчетаPSI

        features_type: str
            Тип признаков в датасете (numeric/categorical)

        Returns:
        --------
        pd.DataFrame
            Датафрейм со значениями PSI на паре наборов данных по
            каждой переменной
        """
        if features_type == "numeric":
            perc = {}
            base_stats = self.cut_buckets_groups(base_df,
                                                 base_df_name, perc=perc)
            diff_stats = self.cut_buckets_groups(diff_df,
                                                 diff_df_name, perc=perc)
        elif features_type == "categorical":
            base_stats = self.calc_buckets_categories(base_df, base_df_name)
            diff_stats = self.calc_buckets_categories(diff_df, diff_df_name)

        all_stats = pd.concat([base_stats.set_index(["feature", "bucket"]),
                               diff_stats.set_index(["feature", "bucket"])],
                              axis=1,
                              join='outer')

        # Заполнить нулями cnt для вновь возникших категорий и
        # оч. маленьким числом share
        all_stats[f"obs_share_{base_df_name}"] = \
            all_stats[f"obs_share_{base_df_name}"].fillna(0.001)
        all_stats[f"obs_share_{diff_df_name}"] = \
            all_stats[f"obs_share_{diff_df_name}"].fillna(0.001)

        all_stats[f"obs_count_{base_df_name}"] = \
            all_stats[f"obs_count_{base_df_name}"].fillna(0)
        all_stats[f"obs_count_{diff_df_name}"] = \
            all_stats[f"obs_count_{diff_df_name}"].fillna(0)

        def _psi(base: pd.Series, diff: pd.Series):
            return (diff - base) * np.log(diff / base)

        all_stats[f"PSI_{base_df_name}_vs_{diff_df_name}"] =\
            _psi(all_stats[f"obs_share_{base_df_name}"],
                 all_stats[f"obs_share_{diff_df_name}"])

        return all_stats

    def calc_psi(self,
                 base_df: pd.DataFrame,
                 base_df_name: str,
                 diff_df: pd.DataFrame,
                 diff_df_name: str) -> pd.DataFrame:
        """
        Общая функция расчета PSI на всем наборе признаков двух наборов данных.
        Управляет запуском расчета PSI для categorical и numeric признаков, а
        также объединением результатов  в один итоговый датафрейм

        Parameters:
        -----------
        base_df: pd.DataFrame
            Датафрейм относительно которого будет считаться PSI

        base_df_name: str
            Имя основного датасета

        diff_df: pd.DataFrame
            Датафрейм на котором будет считаться PSI

        diff_df_name: str
            Имя датасета для расчетаPSI

        Returns:
        --------
        pd.DataFrame:
            Итоговый датафрейм с PSI между двумя наборами данных на всех
            переменных

        """
        
        if self.cat_features is not None:
            numeric = set(self.features_list) - set(self.cat_features)
            categoric = set(self.features_list)\
                .intersection(set(self.cat_features))
        else:
            numeric = set(self.features_list)
            categoric = set([])
            
        numeric.add("y_pred")
        

        if len(numeric) > 0:
            num_stats_psi = self.calc_psi_pair(base_df[numeric]
                                               , base_df_name
                                               , diff_df[numeric]
                                               , diff_df_name
                                               , features_type="numeric")
            total_psi = num_stats_psi.reset_index()
            
        if len(categoric) > 0:
            cat_stats_psi = self.calc_psi_pair(base_df[categoric]
                                               , base_df_name
                                               , diff_df[categoric]
                                               , diff_df_name
                                               , features_type="categorical")

            total_psi = pd.concat([num_stats_psi.reset_index()
                                      , cat_stats_psi.reset_index()]
                                  , axis=0)

        return total_psi

    @calculate_time_execute
    def validate(self, **kwargs):
        """
        Функция инициации расчетов PSI между наборами данных в словаре dict
        Управляет расчетами PSI между парами наборов, объединяет результат в
        итоговый датафрем.
        Записывает результаты проверки PSI и детали расчетов  в excel книгу.

        Parameters:
        -----------
        **kwargs: Dict[str, Tuple(pd.DataFrame, pd.Series)]
            Словарь, где ключ - название датасета, значение -
            кортеж из (X, y), X - матрица признаков,
            y - вектор истинных ответов.
        """
        print("Calculating PSI...")
        # составить чек-лист датафреймов для сравнения с train
        checklist = self._create_checklist(kwargs.keys())

        # считать train
        X_train, y_train = kwargs.get("train", (None, None))
        # нарезать train на event, nevents, all
        train_df = self.create_df(X_train, y_train, self.model,
                                  self.features_list)

        #  Для каждого датафрейма из чек-листа
        for df_name in checklist:

            X_diff, y_diff = kwargs.get(df_name, (None, None))
            # нарезать датафрейм на events, nevents, all
            diff_df = self.create_df(X_diff, y_diff, self.model,
                                     self.features_list)

            _psi_df = pd.DataFrame()
            # для каждой части events, nevents,all посчитать статистики
            for df_part in train_df.keys(): #["all", "events", "nevents"]:
                _psi = self.calc_psi(train_df[df_part], "train",
                                     diff_df[df_part], df_name)
                _psi["data_part"] = df_part
                _psi = _psi.set_index(["feature", "bucket", "data_part"])

                # добавить столбец в датафрейм с итоговым PSI
                _psi_grouped = \
                    _psi.groupby("feature")[f"PSI_train_vs_{df_name}"] \
                    .sum().rename(f"PSI_train_vs_{df_name}_{df_part}")
                self.psi_short = pd.concat([self.psi_short,
                                            _psi_grouped],
                                           axis=1,
                                           sort=True,
                                           join="outer")

                _psi_df = _psi_df.append(_psi)

            new_cols = _psi_df.columns.difference(self.psi_detailed.columns)
            self.psi_detailed = pd.concat([self.psi_detailed,
                                           _psi_df[new_cols]],
                                          axis=1,
                                          join="outer")

        self.psi_detailed = self.psi_detailed.reset_index()
        self.psi_short = self.psi_short.reset_index()
        self.psi_short.rename(columns={"index": "feature"}, inplace=True)

        # Записать результат в excel файл
        # формат
        int_number = "## ##0"
        float_number_high = "## ##0.00"
        float_number_low = "## ##0.0000"
        int_percentage = "0%"
        float_percentage_high = "0.00%"
        float_percentage_low = "0.0000%"

        # Кастомный формат для таблицы
        fmt = {"num_format": {
            int_number: ["obs_count_train",
                         "obs_count_valid",
                         "obs_count_OOT",
                         "obs_count_test",
                         "obs_count_OOT_psi"]
            , float_percentage_low: ["obs_share_train",
                                     "obs_share_valid",
                                     "obs_share_OOT",
                                     "obs_share_test",
                                     "obs_share_OOT_psi"]
            , float_number_low: ["PSI_train_vs_valid",
                                 "PSI_train_vs_test",
                                 "PSI_train_vs_OOT",
                                 "PSI_train_vs_OOT_psi",
                                 "PSI_train_vs_valid_all",
                                 "PSI_train_vs_test_all",
                                 "PSI_train_vs_OOT_all",
                                 "PSI_train_vs_OOT_psi_all",
                                 "PSI_train_vs_valid_events",
                                 "PSI_train_vs_test_events",
                                 "PSI_train_vs_OOT_events",
                                 "PSI_train_vs_OOT_psi_events",
                                 "PSI_train_vs_valid_nevents",
                                 "PSI_train_vs_test_nevents",
                                 "PSI_train_vs_OOT_nevents",
                                 "PSI_train_vs_OOT_psi_nevents"]
             }
        }
        self._to_excel(self.psi_detailed, sheet_name="PSI detailed", fmt=fmt)
        self._to_excel(self.psi_short, sheet_name="PSI", fmt=fmt)
