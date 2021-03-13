import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, log_loss
from sklearn.metrics import roc_auc_score, confusion_matrix
from ..metrics import gini_score, calculate_quantile_bins


class CalculateDataStatistics:
    """
    Расчет статистик по данным. Содержит:

        - статистику по каждой выборке train / valid / ... :
          количество наблюдений в каждой выборке, количество
          целевыйх событий, доля целевого события.

        - статиску по переменным: название целевой переменной,
          количество категориальных признаков, количество непрерывных
          признаков.

        - статистику по переменным: название переменной, количество
          заполненных значений, минимальное значение, среднее значение,
          максимальное значение, перцентили 25, 50, 75.

    Parameters:
    -----------
    models: Dict[string, estiamtor]
        Словарь с моделями: ключ словаря - название модели,
        значение словаря - экземпляр с моделью.

    config: Dict[string, Any]
         Словарь с конфигурацией эксперимента.

    Attributes:
    -----------
    transformer: CategoricalFeaturesTransformer
        Трансформер категориальных признаков.

    gini: pandas.DataFrame
        Датафрейм с анализом переменных по метрике Джини.

    """

    def __init__(self, transformer, gini: pd.DataFrame, config: dict) -> None:
        self.gini = gini
        self.transformer = transformer
        self.categorical = self.transformer.cat_features
        self.config = config

    @staticmethod
    def _calculate_samples_stats(**eval_sets):
        """
        Расчет статистики по выборке data и вектора target.
        Расчитывается количество наблюдений, количество целевых событий
        и доля целевого события.

        Parameters:
        -----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        Returns:
        --------
        result: pandas.DataFrame
            Датафрейм с рассчитанной статистикой.

        """
        result = {}
        for data_name in eval_sets:
            data, target = eval_sets[data_name]
            result[data_name] = [
                len(data), np.sum(target), np.mean(target)
            ]
        result = pd.DataFrame(result).T.reset_index()
        result.columns = ["Выборка", "# наблюдений", "# events", "# eventrate"]
        return result.fillna(0)

    @staticmethod
    def _calculate_variables_stats(**eval_sets) -> pd.DataFrame:
        """
        Расчет статистик по переменным. Рассчитывается количество
        заполненных значений признака, среднее значение признака,
        стандартное отклонение признака, минимальное значение
        признака, 25-ый перцентиль признака, медиана признака,
        75-ый перцентиль признака, максимальное значение признака.

        Parameters:
        -----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        Returns:
        --------
        result: pandas.DataFrame
            Датафрейм с рассчитанной статистикой.


        """
        sample_name = next(iter(eval_sets))
        data, _ = eval_sets[sample_name]

        result = data.describe().T.reset_index()
        result.columns = [
            "Variable name",
            "Number of filled value",
            "AVG-value",
            "STD-value",
            "MIN-value",
            "25% percentile-value",
            "50% percentile-value",
            "75% percentile-value",
            "MAX-value"
        ]
        return result.fillna(0)

    def _calculate_variables_types_stats(self) -> pd.DataFrame:
        """
        Расчет статистик по типам переменным. Рассчитывается количество
        категориальных переменных, количество непрерывных переменных
        и название целевой переменной.

        """
        stats = pd.DataFrame({
            "Целевая переменная": [self.config["target_name"]],
            "# категорий": [len(self.transformer.cat_features)],
            "# непрерывных": [self.gini.shape[0]]
        })
        return stats.fillna(0)

    def transform(self, **eval_sets) -> None:
        """
        Построение отчета с статистиками о данных.

        Parameters:
        -----------
        eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
            Словарь с выборками, для которых требуется рассчитать статистику.
            Ключ словаря - название выборки (train / valid / ...), значение -
            кортеж с матрицей признаков (data) и вектором ответов (target).

        """
        result = (
            self._calculate_samples_stats(**eval_sets),
            self._calculate_variables_types_stats(),
            self._calculate_variables_stats(**eval_sets)
        )
        return result


class CalculateBinaryMetrics:
    """
    Расчет метрик для задачи бинарной классификации:
    GINI, PR-AUC, Log-Loss.

    Parameters
    ----------
    models: dict
        Словарь, ключ - название модели, значение - экземпляр
        ML-модели для DS-Template, из (src.models).

    """
    def __init__(self, models: dict) -> None:
        self.models = models
        self.predictions_ = {}

    @staticmethod
    def create_prediction(model, data: pd.DataFrame) -> np.array:
        """
        Применение модели model к набору данных data.

        Parameters
        ----------
        model: DSPL.models
            Экземпляр ML-модели из DS-Template.

        data: pandas.DataFrame, shape = [n_samples, n_features]
            Набор данных для применения модели.

        Returns
        -------
        pred: np.array
            Вектор прогнозов.

        """
        try:
            pred = model.transform(data)
        except TypeError:
            pred = np.zeros(data.shape[0])

        return pred

    @staticmethod
    def _to_frame(scores: dict, **eval_sets) -> pd.DataFrame:
        """
        Преобразование словаря scores в pandas.DataFrame и
        применение название столбцов (имя метрики + имя выборки).

        Parameters
        ----------
        scores: Dict[string, List[float]]
            Словарь, ключ - название модели, значение - список
            со значениями метрик бинарной классификации.

        eval_sets: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, ключ - название выборки, значение - кортеж
            с матрией признаков и вектором истинных ответов.

        Returns
        -------
        scores: pandas.DataFrame
            Значения метрик.

        """
        scores = pd.DataFrame(scores)
        scores = scores.T.reset_index()

        metrics = ["gini", "pr_auc", "log_loss"]
        scores_name = ["Название модели", "# признаков"]
        scores_name += [f"{metric} {sample}" for metric in metrics
                        for sample in eval_sets]

        scores.columns = scores_name
        scores["детали о модели"] = [
            f"Ссылка на лист train {model}" for model in scores["Название модели"]]
        return scores

    def create_all_predictions(self, **eval_set):
        """
        Применение всех моделей из self.models к наборам данных
        из eval_set
        """
        for model in self.models:
            self.predictions_[model] = {}
            sample_pred = {}

            for sample in eval_set:
                data, _ = eval_set[sample]
                pred = self.create_prediction(self.models[model], data)
                sample_pred[sample] = pred

            self.predictions_[model] = sample_pred

    def calculate_metrics(self, model_name, **kwargs):
        """
        Вычисление метрик GINI, PR-AUC, Log-Loss.

        Parameters
        ----------
        model_name: string
            Название модели из self.models.

        kwargs: dict
            Словарь, ключ - название выборки, значение - кортеж
            с матрицой признаков для применения модели и вектором
            истинных ответов.

        Returns
        -------
        metrics_score: list
            Список со значением метрик.

        """
        try:
            model = self.models[model_name]
            metrics_score = [len(model.used_features)]
        except TypeError:
            sample_name = next(iter(kwargs))
            metrics_score = [len(kwargs[sample_name][0])]

        for metric in [gini_score, average_precision_score, log_loss]:

            for sample in kwargs:

                _, target = kwargs[sample]
                pred = self.predictions_[model_name]
                try:
                    metrics_score.append(round(100*metric(target, pred[sample]), 4))
                except ValueError:
                    metrics_score.append(0)

        return metrics_score

    def transform(self, **eval_sets):
        """
        Расчет метрик бинарной классификации для
        каждой модели из self.models и каждой выборки из
        eval_sets.

        Parameters
        ----------
        eval_sets: Dict[string, Tuple[pd.DataFrame, pd.Series]]
            Словарь, ключ - название выборки, значение - кортеж
            с матрией признаков и вектором истинных ответов.

        Returns
        -------
        scores: pandas.DataFrame
            Значения метрик.

        """
        scores = {}
        self.create_all_predictions(**eval_sets)

        for model in self.models:
            scores[model] = self.calculate_metrics(model, **eval_sets)

        return self._to_frame(scores, **eval_sets)


class CalculateDetailedMetrics:
    """
    Расчет детальных метрик для задачи бинарной классификации.

    Рассчитываются метрики по бинам прогнозных значений модели.
    Для каждого бина рассчитывется:
        - минимальная вероятность в бине;
        - средняя вероятность в бине;
        - максимальная вероятность в бине;
        - доля целевого события в бине (evenrate);
        - количество наблюдений в бине;
        - количество целевых событий в бине;
        - количество нецелевых событий в бине;
        - кумулятивное количество целевых событий в бине;
        - кумулятивное количество нецелевых событий в бине;
        - FPR в бине;
        - TPR в бине;
        - GINI на данном выборке;
        - ROC-AUC на данной выборке;
        - стандартная ошибка ROC-AUC;
        - 95% доверительный интервал метрики ROC-AUC.

    Parameters
    ----------
    n_bins: integer, optional, default = 20
        Количество бинов.

    """
    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins

    @staticmethod
    def calculate_conf_interval(data: pd.DataFrame,
                                scores: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет доверительного интервала и стандартной ошибки ROC AUC.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, 3]
            Датафрейм с прогнозами модели (y_pred),
            истинными значениями целевой переменной (y_true)
            и рассчитанным бином (bin).

        scores: pandas.DataFrame, shape = [self.n_bins, ]
            Датафрейм с расчетом базовым метрик по бинам.

        Returns
        -------
        scores: pandas.DataFrame
            Датафрейм с рассчитаными метриками по бинам,
            доверительным интервалом ROC AUC и стандартной
            ошибкой ROC AUC.

        """
        num_row = scores.shape[0]
        auc = 100 * roc_auc_score(data["y_true"], data["y_pred"])
        gini = 2 * auc - 100

        std_error = 1.96 * np.sqrt(auc*(100 - auc)/data.shape[0])
        scores.loc[num_row, "GINI"] = gini
        scores.loc[num_row, "AUC"] = auc
        scores.loc[num_row, "AUC Std Err"] = std_error
        scores.loc[num_row, "AUC 95% LCL"] = auc - std_error
        scores.loc[num_row, "AUC 95% UCL"] = auc + std_error

        return np.round(scores, 4)

    @staticmethod
    def calculate_total_stats(scores: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет базовых метрик по всему набору данных.
        Базовые метрики: общее число наблюдений в выборке, количество
        целевых событий, количество нецелевых событий, доля целевого
        события в выборке.

        Parameters
        ----------
        scores: pandas.DataFrame
            Датафрейм с расчитанными базовыми метрик по бинам.

        Returns
        -------
        scores: pandas.DataFrame
            Датафрейс с расчитанными базовыми метриками по бинам
            и базовыми метрики по всему набору данных.

        """
        num_row = scores.shape[0] - 1
        scores.loc[num_row, "bin"] = "Total"
        scores.loc[num_row, "#obs"] = scores["#obs"].sum()
        scores.loc[num_row, "#event"] = scores["#event"].sum()
        scores.loc[num_row, "#nonevent"] = scores["#nonevent"].sum()
        scores.loc[num_row, "eventrate"] = scores["#event"].sum() / scores["#obs"].sum()
        return scores

    def calculate_base_bins_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет метрик по бинам.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, 3]
            Датафрейм с прогнозами модели (y_pred),
            истинными значениями целевой переменной (y_true)
            и рассчитанным бином (bin).

        Returns
        -------
        scores: pandas.DataFrame, shape = [self.n_bins, ]
            Датафрейм с метриками по бинам.

        """
        data_gp = data.groupby(["bin"])
        scores = data_gp.agg({
            "y_pred": ["min", "mean", "max"],
            "y_true": ["mean", "count", "sum"]
        })
        scores.columns = [
            "prob_min", "prob_mean", "prob_max", "eventrate", "#obs", "#event"]
        scores = scores.reset_index()
        scores["bin"] = np.arange(1, scores.shape[0] + 1)
        scores = scores.sort_values(by="bin", ascending=False)

        scores["#nonevent"] = scores["#obs"] - scores["#event"]
        scores["cum # ev"] = scores["#event"].cumsum()
        scores["cum # nonev"] = scores["#nonevent"].cumsum()
        scores = scores.reset_index(drop=True)

        try:
            fpr, tpr = self.roc_auc_metrics(data)
            tpr = np.round(100 * tpr, 4)
            fpr = np.round(100 * fpr, 4)
            scores["1 - Specificty"] = fpr[:len(scores)][::-1]
            scores["Sensitivity"] = tpr[:len(scores)][::-1]
            return scores
        except ValueError:
            return scores

    def roc_auc_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет метрик FPR, TPR по бинам.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, 3]
            Датафрейм с прогнозами модели (y_pred),
            истинными значениями целевой переменной (y_true)
            и рассчитанным бином (bin).

        Returns
        -------
        sensitivity_score, specificity_score: numpy.array
            Метрики FPR, TPR по бинам.

        """
        sensitivity_score = np.zeros(self.n_bins)
        specificity_score = np.zeros(self.n_bins)
        unique_bins = data["bin"].value_counts()
        sorted_bins = unique_bins.sort_index().index

        for num, bin_ in enumerate(sorted_bins):
            mask = data["bin"] == bin_
            threshold = data.loc[mask, "y_pred"].min()
            y_pred_labels = np.where(data["y_pred"] >= threshold, 1, 0)
            tn, fp, fn, tp = confusion_matrix(data.y_true, y_pred_labels).ravel()
            sensitivity_score[num] = (tp / (tp + fn))
            specificity_score[num] = (fp / (fp + tn))

        return sensitivity_score, specificity_score

    def transform(self, y_true, y_pred):
        """
        Расчет метрик для каждого бина.

        Parameters
        ----------
        y_true: array-like, shape = [n_samples, ]
            Вектор истинных ответов.

        y_pred: array-like, shape = [n_samples, ]
            Вектор прогнозов.

        Returns
        -------
        scores: pandas.DataFrame, shape = [self.n_bins, 17]
            Датафрейм с рассчитаными по бинам метриками.

        """
        data = pd.DataFrame({"y_pred": y_pred, "y_true": y_true})
        data["bin"] = calculate_quantile_bins(data["y_pred"], self.n_bins)
        data["bin"] = data["bin"].fillna(-1)

        scores = self.calculate_base_bins_metrics(data)
        scores = self.calculate_conf_interval(data, scores)
        scores = self.calculate_total_stats(scores)
        return scores.fillna(".")
