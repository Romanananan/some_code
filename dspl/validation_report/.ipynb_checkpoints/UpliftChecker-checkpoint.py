from copy import deepcopy
from tqdm import tqdm

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

from .Checker import Checker
from .FormatExcelWriter import FormatExcelWriter

from .funcs import calculate_permutation_feature_importance

class UpliftChecker(Checker):

    def __init__(self,
                 writer: pd.ExcelWriter,
                 model_name: str,
                 model,
                 features_list: list,
                 model_type: str = None,
                 cat_features: list = None,
                 drop_features: list = None,
                 current_path: str = None,
                 handbook=None):

        self.calc_dataset = "train"
        self.model_name = model_name
        self.writer = writer
        self.model = deepcopy(model)
        self.features_list = features_list
        self.model_type = model_type
        self.cat_features = cat_features
        self.drop_features = drop_features
        self.current_path = current_path
        self.handbook = handbook

    def plot_uplift_curve(self, uplift_df: pd.DataFrame, plot_name:str = None):

        fig, ax1 = plt.subplots(figsize=(12, 10))

        ax1.plot(uplift_df["feature_num"], uplift_df[f"{self.metric_name}_train"],
                 linewidth=3, marker="o")
        ax1.plot(uplift_df["feature_num"], uplift_df[f"{self.metric_name}_valid"],
                 linewidth=3, marker="o")
        ax1.legend([f"{self.metric_name} train", f"{self.metric_name} valid"])
        ax1.grid()

        ax1.axvline(int(0.6 * len(uplift_df)), linestyle="dashed",
                    color="gray")
        ax1.axhline(uplift_df[f"{self.metric_name}_valid"].max() - 0.01,
                    linestyle="dashed", color="gray")

        ax1.set_xlabel("number of features")
        ax1.set_ylabel(f"{self.metric_name}")

        ax1.annotate(f"maximum {self.metric_name} -1pp",
                     xy=(0, uplift_df[f"{self.metric_name}_valid"].max() - 0.01),
                     xycoords="data")

        ax1.annotate("60% of features",
                     xy=(int(len(uplift_df) * 0.6), 0.15),
                     xycoords="data")
        fig.tight_layout()

        if plot_name is not None:
            plt.savefig(f"{plot_name}.png", bbox_inches="tight")

    def _metric(self, y_true, y_pred):
        """
        Метрика качества для сравнения модели на усеченном списке атрибутов
        Бинарная классификация - Джини
        Регресиия - среднее значение абсолютного отклонения
        """
        if self.model_type == "binary_classification":
            return 2*roc_auc_score(y_true, y_pred) - 1

        elif self.model_type == "regression":
            return spearmanr(y_true, y_pred)[0]

    def _metric_func(self):

        if self.model_type == "binary_classification":
            return roc_auc_score

        elif self.model_type == "regression":
            return spearmanr
        else:
            raise ValueError("model type can be 'binary_classification' or"
                             " 'regression' ")

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

        float_percentage_high = "0.00%"
        float_number_low = "## ##0.0000"
        fmt = {"num_format": {
            float_percentage_high: [f"{self.metric_name}_train",
                                    f"{self.metric_name}_valid",
                                    "features_share", "delta_full"] }
        }

        excelWriter = FormatExcelWriter(self.writer)
        excelWriter.write_data_frame(df, (0, 0), sheet=sheet_name,
                                     formats=fmt)

        # yellow rows
        yellow = df.index[(df["delta_full"] < 0.01)
                          & (df["features_share"] <= 0.6)].to_list()
        for row in yellow:
            excelWriter.set_cell_cond_format(df, (0, 0), "delta_full", row,
                                             -1, 0.01, order="reverse")

        # red validation
        red = df.index[(df["delta_full"] <= -0.05)].to_list()

        for row in red:
            excelWriter.set_cell_cond_format(df, (0, 0), "delta_full", row,
                                             -0.01, 1, order="reverse")

        if plot:
            sheet = self.writer.sheets[sheet_name]
            sheet.insert_image(
                f"A{df.shape[0]+3}", f"{self.current_path}/images/{sheet_name}.png")

        # Доп инфо
        sheet = self.writer.sheets[sheet_name]
        sheet.write_string(f"H4",
                           f"* - Красный светофор - существует набор факторов "
                           f"на которых метрика качества на 5% выше, чем "
                           f"на полном наборе. {chr(10)}")

        sheet.write_string(f"H5",
                           f"* - Желтый светофор - существует набор не более "
                           f"60% факторов, значение метрики качества на "
                           f"котором уступает не более чем на 1% качеству "
                           f"на полном наборе")

    def get_abstract_model_copy(self):
        model_params = self.model.get_params()
        model_class = self.model.__class__

        new_model = model_class(**model_params)

        return new_model

    def _get_alt_pred(self, model, features):

        if hasattr(model,"predict_proba"):
            y_pred = model.predict_proba(features)[:,1]
        elif hasattr(model, "predict"):
            y_pred = model.predict(features)
        elif hasattr(model, "transform"):
            y_pred = model.transform(features)
        else:
            raise AttributeError("No prediction method provided")

        return y_pred

    def get_abstract_uplift_scores(self, **kwargs):

        scores_train = []
        scores_valid = []

        x_train, y_train = kwargs["train"]
        x_valid, y_valid = kwargs["test"]

        features_imp = calculate_permutation_feature_importance(
            self.model,
            self._metric_func(),
            y_train,
            x_train[self.features_list])["feature"].to_list()

        for feature_num, _ in enumerate(tqdm(features_imp)):
            features = features_imp[:feature_num + 1]
            print(features)
            alt_model = self.get_abstract_model_copy()
            alt_model.fit(x_train[features], y_train)

            y_pred_train = self._get_alt_pred(alt_model, x_train[features])
            scores_train.append(self._metric(y_train, y_pred_train))

            y_pred_valid = self._get_alt_pred(alt_model, x_valid[features])
            scores_valid.append(self._metric(y_valid, y_pred_valid))

            # print score result
            # alt_model.evaluate_model(**kwargs)


        scores_df = pd.DataFrame({"feature": features_imp,
                                  f"{self.metric_name}_train": scores_train,
                                  f"{self.metric_name}_valid": scores_valid})

        feature_num = np.arange(len(scores_df)) + 1
        scores_df["feature_num"] = feature_num
        scores_df["features_share"] = feature_num / feature_num.max()
        scores_df["delta_full"] = scores_df[f"{self.metric_name}_valid"].iloc[-1] - \
                          scores_df[f"{self.metric_name}_valid"]

        return scores_df


    def get_template_model_copy(self, features_list):

        # get params for creating estimator
        model_obj_params = self.model.get_params()

        # parse params
        params = model_obj_params.pop("params")
        model_obj_params.pop("used_features")
        optional_params = model_obj_params

        model_class = self.model.__class__

        # TODO: params-> **params or not?
        new_model = model_class(params, features_list, **optional_params)

        return new_model

    def get_template_uplift_scores(self, **kwargs):

        # get model features to iterate through
        features_imp = list(
            self.model.feature_importance(*kwargs["train"])["feature"])

        scores_train = []
        scores_valid = []

        x_train, y_train = kwargs["train"]
        x_valid, y_valid = kwargs["valid"]

        for feature_num, _ in enumerate(tqdm(features_imp)):
            features = features_imp[:feature_num + 1]
            
            alt_model = self.get_template_model_copy(features)
            alt_model.fit(x_train, y_train, x_valid, y_valid)

            y_pred_train = alt_model.transform(x_train)
            scores_train.append(self._metric(y_train, y_pred_train))
            
            y_pred_valid = alt_model.transform(x_valid)
            scores_valid.append(self._metric(y_valid, y_pred_valid))

#             print(features)
#             alt_model.evaluate_model(**kwargs)

        scores_df = pd.DataFrame({"feature": features_imp,
                                  f"{self.metric_name}_train": scores_train,
                                  f"{self.metric_name}_valid": scores_valid})

        feature_num = np.arange(len(scores_df)) + 1
        scores_df["feature_num"] = feature_num
        scores_df["features_share"] = feature_num / feature_num.max()
        scores_df["delta_full"] = scores_df[f"{self.metric_name}_valid"].iloc[-1] - \
            scores_df[f"{self.metric_name}_valid"]

        return scores_df

    def get_subset(self, **kwargs):

        val_set = {}

        if "train" in kwargs:
            x_train, y_train = kwargs["train"]
            if x_train.shape[0] > 200000:
                val_set["train"] = (x_train.sample(n=200000, axis=0,
                                                   random_state=17),
                                    y_train.sample(n=200000, random_state=17))
            else:
                val_set["train"] = (x_train, y_train)
        else:
            raise KeyError("There must be train dataset provided")

        if "valid" in kwargs:
            x_valid, y_valid = kwargs["valid"]
            if x_valid.shape[0] > 200000:
                val_set["valid"] = (x_valid.sample(n=200000, axis=0,
                                                   random_state=17),
                                    y_valid.sample(n=200000, random_state=17))
            else:
                val_set["valid"] = (x_valid, y_valid)
        elif "test" in kwargs:
            x_test, y_test = kwargs["test"]
            if x_test.shape[0] > 200000:
                val_set["test"] = (x_test.sample(n=200000, axis=0,
                                                   random_state=17),
                                   y_test.sample(n=200000, random_state=17))
            else:
                val_set["test"] = (x_test, y_test)
        else:
            raise KeyError("No test-valid dataset is provided")

        return val_set

    def get_features_subset(self):
        features_subsets = []

        if len(self.features_list) > 20:
            for it in range(5, len(self.features_list)+5, 5):
                features_subsets.append(self.features_list[:it])
            features_subsets.append(self.features_list)
        else:
            features_subsets = self.features_list

        return features_subsets

    @property
    def metric_name(self):

        if self.model_type == "binary_classification":
            metric = "gini"
        elif self.model_type == "regression":
            metric = "Spearman correlation"

        return metric

    def validate(self, excel: bool = True, **kwargs):

        # generate dataset for test
        data_sets = self.get_subset(**kwargs)

        if self.model.__module__.split(".")[0] == "dspl":
            # ds template module
            uplift_scores = self.get_template_uplift_scores(**data_sets)
        else:
            # abstract model
            uplift_scores = self.get_abstract_uplift_scores(**data_sets)
        if excel:
            plot_path = f"{self.current_path}/images/uplift_test"
            self.plot_uplift_curve(uplift_scores, plot_name=plot_path)
            self._to_excel(uplift_scores, sheet_name="uplift_test", plot=True)
        else:
            return uplift_scores
