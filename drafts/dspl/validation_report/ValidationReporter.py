import os
import pandas as pd
from copy import deepcopy
from .CreateReport import CreateReport
from ..utils import calculate_time_execute


class ValidationReporter:
    """
    Класс управления вызовами проверок, порядком вызова и набора тестов.
    Парсит конфигурационный файл, создает и сохраняет файл валидационного
    отчета.

    Parameters:
    -----------
        models: dict
            Словарь моделей вида: {<имя модели>:
                                   (<объект модели>, <список фичей>)}

        config: dict
            Конфигурационный файл с параметрами запуска

    Attributes:
    ------------
        self.models: dict
            Копия словаря моделей для построения отчета

        self.writer: xlsxWriter
            Указатель на созданный файл для записи отчета

        self.current_path
            Путь к текущей директории

        self.checklist: list
            Список проверок, которые будут выполнены в ходе запуска

        self.categorical_features: list
            Список категориальных переменных

        self.drop_features: list
            Список мусорных или технических фичей

        self.hdbk: pd.DataFrame
            Ручной справочник с расшифровкой имен переменных
    """

    def __init__(self,
                 models: dict,
                 names: list = None,
                 config: dict = None,
                 current_path: str = None,
                 uplift_builder: bool = True
                 ) -> None:

        self.models = deepcopy(models)
        self.writer = None
        self.current_path = current_path
        self.uplift_builder = uplift_builder

        # чтени доп. объектов из шаблона разработки
        try:
            self.encoder = self.models.pop("encoder")
        except KeyError:
            self.encoder = None

        if config.get("log_target", None):
            self.log_transformer = self.models.pop("log_target_transformer")
        else:
            self.log_transformer = None

        #  оставить только модели из списка на проверку
        if names is not None:
            self.models = {name: model for (name, model) in self.models.items()
                           if name in names}

        # обновление списка кат. фичей из encoder
        if self.encoder is not None:
            self.categorical_features = self.encoder.cat_features
        else:
            self.categorical_features = config.get("categorical_features",
                                                   None)

        self.drop_features = config.get("drop_features", None)

        self.model_type = config.get("model_type", None)

        if self.model_type == "binary_classification":
            self.checklist = ["data_stats",
                              "Gini_vars",
                              "models_report",
                              "Gini_stability",
                              "Permutation_imp",
                              "PSI",
                             ]
            if self.uplift_builder:
                self.checklist.append("Uplift")
        elif self.model_type == "regression":
            self.checklist = ["data_stats",
                              "TreeCorr",
                              "models_report",
                              "RegMetrics",
                              "REC",
                              "Permutation_imp",
                              "PSI"
                              ]
        else:
            self.checklist = ["data_stats",
                              "Gini_vars",
                              "models_report",
                              "Gini_stability",
                              "Permutation_imp",
                              "SHAP",
                              "PSI"]

        self.hdbk_path = config.get("variables_description", None)
        if self.hdbk_path is not None:
            try:
                self.hdbk = pd.read_excel(self.hdbk_path).set_index("variable")
                print("Variable description has been read successfully")
            except:
                self.hdbk = None
                print("Variables description has not been read")
        else:
            self.hdbk = None

        self._set_working_dir(config)

    def _create_sub_dir(self):
        """
        Создание под-директорий для сохранения отчетов.
        Создаются директории: docs - для сохранение отчета
        о моделях, images - сохранение графиков
        """
        os.mkdir(f"{self.current_path}/docs/")
        os.mkdir(f"{self.current_path}/images/")

    def _set_working_dir(self, config: dict) -> None:
        """
        Определение рабочей директории для сохранения файлов отчета
        устанавливает current_path в зависимости от config.

        Полагается, что если присутствует конфигурация model_path - модуль
        работает в режиме создания отчета для самостоятельно разработанной
        модели.

        Parameters:
        -----------
            config: dict
                Словарь с конфигурациями запуска
        """
        if self.current_path is None:
            try:
                # список папок с номерами запусков
                dirs = [
                    int(num) for num in os.listdir("runs/")
                    if not num.startswith(".")]

                if config.get("model_path", None) is None:
                    # Определить последний номер запуска разработки модели
                    self.current_path = f"runs/{max(dirs)}"
                else:
                    # Создать рабочие директории для отчета
                    self.current_path = f"runs/{max(dirs) + 1}"
                    os.mkdir(self.current_path)
                    self._create_sub_dir()

            except FileNotFoundError:
                os.mkdir("runs")
                os.mkdir("runs/1")
                self.current_path = "runs/1"
                self._create_sub_dir()
        else:
            try:
                self._create_sub_dir()
            except FileExistsError:
                pass

    @calculate_time_execute
    def transform(self, **kwargs):
        """
        Запуск поочередного выполнения валидационных тестов.
        Предварительно создаются директории для хранения отчетов и файл
        для записи результата каждго теста

        Parameters:
        -----------
        **kwargs:
            Список параметров пере длины. На вход ожидается в виде словаря
            dict  с датасетами упакованными в виду следующих записей:
            {<dataset_name> : tuple(X,y) }
        """
        """
        # автоматическое добавление категориальных фичей
        x_train, y_train = kwargs.get("train", None)
        categorical = x_train.select_dtypes(include=["object", "category"])\
            .columns().tolist()

        self.categorical_features = set(self.categorical_features)\
                                    + set(categorical)  
        """
        x_train, _ = kwargs.get("train", (None, None))
        categories = x_train.select_dtypes(include=["object", "category"])\
                    .columns.tolist()

        if self.categorical_features is not None:
            self.categorical_features = list(set(self.categorical_features\
                                                 + categories))
        else:
            self.categorical_features = categories

        for model_name, mdl in self.models.items():
            if isinstance(mdl, tuple):
                model, features_list = mdl
            else:
                model = mdl
                features_list = mdl.used_features

            writer = pd.ExcelWriter(path=f"{self.current_path}"
                                         f"/docs/Validation_report"
                                         f"_{model_name}.xlsx")
            creator = CreateReport(writer=writer,
                                   model_name=model_name,
                                   model=model,
                                   features_list=features_list,
                                   cat_features=self.categorical_features,
                                   drop_features=self.drop_features,
                                   model_type=self.model_type,
                                   target_transformer=self.log_transformer,
                                   current_path=self.current_path,
                                   handbook=self.hdbk)

            for test in self.checklist:
                reporter = creator.create_reporter(test)
                reporter.validate(**kwargs)         
            
            writer.save()
            #print("Total report creating time: ")
