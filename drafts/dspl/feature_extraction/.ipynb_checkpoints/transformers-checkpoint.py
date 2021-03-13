import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional, List
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from .utils import find_categorical_features
from ..utils import MissedColumnError


class LogTargetTransformer(BaseEstimator, TransformerMixin):
    """
    Преобразование целевой переменной в логарифмированную шкалу.

    Parameters
    ----------
    bias: float, optional, default = 0
        Смещение, добавляется к аргументу логарифма.
        Опциональный параметр, по умолчанию равен 0.

    tolerance: float, optional, default = 1e-5
        Значение, добавляемое к аргументу логарифма, для того,
        чтобы избежать ситуаций np.log(0).

    Attributes
    ----------
    target_min: float
        Минимальное значение целевой переменной.

    fitted: bool
        Флаг применения метода fit к целевой переменной.

    """
    def __init__(self, bias: float = 0, tolerance: float = 1e-5):
        self.bias = bias
        self.tolerance = tolerance
        self.target_min = None
        self.fitted = None

    @property
    def check_is_fitted(self):
        if not self.fitted:
            msg = ("This estimator is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    def fit(self, target: pd.Series) -> None:
        """
        Расчет минимального значения целевой переменной для
        корректного расчета логарифма на отрицательных значениях.

        Parameters
        ----------
        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        self

        """
        self.target_min = target.min()
        self.fitted = True
        return self

    def transform(self, target: pd.Series) -> pd.Series:
        """
        Логарифмическое преобразование целевой переменной.

        Parameters
        ----------
        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        log_target: pandas.Series, shape = [n_samples, ]
            Вектор прологарифмированной целевой переменной.

        """
        self.check_is_fitted
        self.target_min = target.min()
        return np.log(target - self.target_min + 1 + self.bias)

    def inverse_transform(self, target: pd.Series) -> pd.Series:
        """
        Преобразование прологарифмированной целевой переменной к
        значениям в исходном диапазоне.

        Parameters
        ----------
        log_target: pandas.Series, shape = [n_samples, ]
            Вектор прологарифмированной целевой переменной.

        Returns
        -------
        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        """
        self.check_is_fitted
        return np.exp(target) + self.target_min - self.tolerance - 1


class PSI(BaseEstimator, TransformerMixin):
    """
    Вычисление PSI и отбор признаков на их основе.

    Parameters
    ----------
    threshold: float
        Порог для отбора переменных по PSI.
        Если PSI для переменной выше порога - переменная макрируется
        0 (не использовать для дальнейшего анализа), если ниже
        порога - маркируется 1 (использовать для дальнейшего анализа).

    n_bins: int, optional, default = 20
        Количество бинов, на которые разбивается выборка.

    min_value: float, optional, default = 0.005
        Значение которое используется, если рассчитанный psi = 0.

    bin_type: string, optional, default = "quanitles"
        Способ разбиения на бины: "quantiles" or "bins".
        При выборе "quantiles" - выборка будет разбита на n_bins
        квантилей, при выборке "bins" - выборка будет разбита на
        n_bins бакетов с равным шагом между бакетами.
        Иные значения приводят к возникновению ValueError.

    Attributes
    ----------
    scores_: Dict[str, float]
        Словарь со значениями PSI,
        ключ словаря - название признака, значение - PSI-score.

    """
    def __init__(self,
                 threshold: float,
                 categorical_features: Optional[List[str]] = None,
                 bin_type: str = "quantiles",
                 min_value: float = 0.005,
                 n_bins: int = 20):

        self.threshold = threshold
        self.categorical_features = categorical_features
        self.min_value = min_value
        self.n_bins = n_bins
        if bin_type in ["quantiles", "bins"]:
            self.bin_type = bin_type
        else:
            raise ValueError(
                "Incorrect bin_type value. Expected 'quantiles' or 'bins', "
                f"but {bin_type} is transferred."
            )
        self.scores = {}

    @property
    def check_is_fitted(self):
        if not self.scores:
            msg = ("This estimator is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    def calculate_bins(self, data: pd.Series) -> np.array:
        """
        Вычисление границ бинов для разбиения выборки.

        Parameters
        ----------
        data: pandas.Series, shape = [n_samples, ]
            наблюдения из train-выборки.

        Returns
        -------
        bins: numpy.array, shape = [self.n_bins + 1]
            Список с границами бинов.

        """
        if self.bin_type == "quantiles":
            bins = np.linspace(0, 100, self.n_bins + 1)
            bins = [np.nanpercentile(data, x) for x in bins]

        else:
            bins = np.linspace(data.min(), data.max(), self.n_bins + 1)

        return np.unique(bins)

    def calculate_psi_in_bin(self, expected_score, actual_score) -> float:
        """
        Вычисление значения psi для одного бакета.

        Осуществляется проверка на равенство нулю expected_score и
        actual_score: если один из аргументов равен нулю, то его
        значение заменяется на self.min_value.

        Parameters
        ----------
        expected_score: float
            Ожидаемое значение.

        actual_score: float
            Наблюдаемое значение.

        Returns
        -------
        value: float
            Значение psi в бине.

        """
        if expected_score == 0:
            expected_score = self.min_value
        if actual_score == 0:
            actual_score = self.min_value

        value = (expected_score - actual_score)
        value = value * np.log(expected_score / actual_score)

        return value

    def calculate_psi(self, expected: pd.Series, actual: pd.Series, bins) -> float:
        """
        Расчет PSI для одной переменной.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        bins: pandas.Series, shape = [self.n_bins, ]
            Бины для расчета PSI.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        expected_score = np.histogram(expected.fillna(-9999), bins)[0]
        expected_score = expected_score / expected.shape[0]

        actual_score = np.histogram(actual.fillna(-9999), bins)[0]
        actual_score = actual_score / actual.shape[0]

        psi_score = np.sum(
            self.calculate_psi_in_bin(exp_score, act_score)
            for exp_score, act_score in zip(expected_score, actual_score)
        )

        return psi_score

    def calculate_numeric_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """
        Вычисление PSI для числовой переменной.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        bins = self.calculate_bins(expected)
        psi_score = self.calculate_psi(expected, actual, bins)
        return psi_score

    def calculate_categorical_psi(self, expected: pd.Series, actual: pd.Series) -> float:
        """
        Вычисление PSI для категориальной переменной.
        PSI рассчитывается для каждого уникального значения категории.

        Parameters
        ----------
        expected: pandas.Series, shape = [n_samples_e, ]
            Наблюдения из train-выборки.

        actual: pandas.Series, shape = [n_samples_o, ]
            Наблюдения из test-выборки.

        Returns
        -------
        psi_score: float
            PSI-значение для данной пары выборок.

        """
        bins = np.unique(expected).tolist()
        expected_score = expected.value_counts(normalize=True)
        actual_score = actual.value_counts(normalize=True)

        expected_score = expected_score.sort_index().values
        actual_score = actual_score.sort_index().values

        psi_score = np.sum(
            self.calculate_psi_in_bin(exp_score, act_score)
            for exp_score, act_score in zip(expected_score, actual_score)
        )
        return psi_score

    def fit(self, data, target=None):
        """
        Вычисление PSI-значения для всех признаков.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        self

        """
        missed_columns = list(set(data.columns) - set(target.columns))

        if missed_columns:
            raise MissedColumnError(
                f"Missed {list(missed_columns)} columns in data.")

        if self.categorical_features:
            numeric_features = list(
                set(data.columns) - set(self.categorical_features)
            )
            self.categorical_features = list(
                set(data.columns) & set(self.categorical_features)
            )
            for feature in self.categorical_features:
                self.scores[feature] = self.calculate_categorical_psi(
                    data[feature], target[feature]
                )
        else:
            numeric_features = data.columns

        for feature in tqdm(numeric_features, leave=False):
            self.scores[feature] = self.calculate_numeric_psi(
                data[feature], target[feature]
            )
        return self

    def transform(self, data, target=None) -> pd.DataFrame:
        """
        Отбор переменных по self.threshold.
        Если PSI-score для переменной выше порога, то переменная
        помечается 0 (не использовать для дальнейшего анализа), если ниже
        порога - маркируется 1 (использовать для дальнейшего анализа).

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        scores: pandas.DataFrame, shape = [n_features, 3]
            Датафрейм с PSI-анализом переменных.

        """
        self.check_is_fitted
        scores = pd.Series(self.scores)
        scores = pd.DataFrame({"Variable": scores.index, "PSI": scores.values})
        scores["Selected"] = np.where(scores.PSI < self.threshold, 1, 0)
        scores = scores.sort_values(by="PSI")

        mask = scores["Selected"] == 1
        self.used_features = scores.loc[mask, "Variable"].tolist()

        return scores.reset_index(drop=True)


class GiniFeatureImportance(BaseEstimator, TransformerMixin):
    """
    Вычисление GINI для каждого признака и отбор признаков на их основе.

    Parameters
    ----------
    threshold: float
        Порог для отбора переменных по PSI.
        Если PSI для переменной выше порога - переменная макрируется
        0 (не использовать для дальнейшего анализа), если ниже
        порога - маркируется 1 (использовать для дальнейшего анализа).

    n_bins: int, optional, default = 20
        Количество бинов, на которые разбивается выборка.

    min_value: float, optional, default = 0.005
        Значение которое используется, если рассчитанный psi = 0.

    bin_type: string, optional, default = "quanitles"
        Способ разбиения на бины: "quantiles" or "bins".
        При выборе "quantiles" - выборка будет разбита на n_bins
        квантилей, при выборке "bins" - выборка будет разбита на
        n_bins бакетов с равным шагом между бакетами.
        Иные значения приводят к возникновению ValueError.

    Attributes
    ----------
    scores_: Dict[str, float]
        Словарь со значениями PSI,
        ключ словаря - название признака, значение - PSI-score.

    """

    def __init__(self, threshold, cat_features=None):
        self.threshold = threshold
        self.cat_features = cat_features
        self.used_features = None
        self.scores = {}

    @property
    def check_is_fitted(self):
        if not self.scores:
            msg = ("This estimator is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    def fit(self, data, target):
        """
        Вычисление метрики GINI для всех признаков.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        self

        """
        if self.cat_features:

            missed_columns = list(
                set(self.cat_features) - set(data.columns)
            )
            if missed_columns:
                raise MissedColumnError(
                    f"Missed {list(missed_columns)} columns in data.")

            numeric_features = list(
                set(data.columns) - set(self.cat_features)
            )
        else:
            numeric_features = data.columns

        for feature in tqdm(numeric_features, leave=False):
            auc = roc_auc_score(target, data[feature].fillna(-9999))
            self.scores[feature] = (2*np.abs(auc - 0.5)) * 100

        return self

    def transform(self, data, target=None):
        """
        Отбор переменных по self.threshold.
        Если GINI для переменной выше порога, то переменная
        помечается 1 (использовать для дальнейшего анализа), если ниже
        порога - маркируется 0 (не использовать для дальнейшего анализа).

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для тестирования.

        Returns
        -------
        scores: pandas.DataFrame, shape = [n_features, 3]
            Датафрейм с GINI-анализом переменных.

        """
        self.check_is_fitted
        scores = pd.Series(self.scores)
        scores = pd.DataFrame({"Variable": scores.index, "GINI": scores.values})
        scores["Selected"] = np.where(scores.GINI > self.threshold, 1, 0)
        scores = scores.sort_values(by="GINI", ascending=False)

        if self.cat_features:
            cat_features_scores = pd.DataFrame({
                "Variable": self.cat_features,
                "GINI": "категориальный признак",
                "Selected": 1
            })
            scores = scores.append(cat_features_scores)

        mask = scores["Selected"] == 1
        self.used_features = scores.loc[mask, "Variable"].tolist()

        return scores.reset_index(drop=True)


class DecisionTreeFeatureImportance(BaseEstimator, TransformerMixin):
    """
    Отбор признаков на основе решающего дерева.

    Parameters
    ----------
    threshold: float
        Порог для отбора переменных по корреляции.
        Если коэффициент корреляции для переменной выше 
        порога - переменная макрируется 1 (использовать для 
        дальнейшего анализа), если ниже порога - маркируется 0 
        (не использовать для дальнейшего анализа).

    cat_features: List[string], optional, default = None
        Список категориальных признаков. 
        Опциональный параметр, по умолчанию, не используется.

    Attributes
    ----------
    scores_: Dict[str, float]
        Словарь со значениями корреляции,
        ключ словаря - название признака, значение - correlation-score.

    """
    def __init__(self, threshold, cat_features=None):
        self.threshold = threshold
        self.cat_features = cat_features
        self.scores = {}

    @property
    def check_is_fitted(self):
        if not self.scores:
            msg = ("This estimator is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    @staticmethod
    def calculate_tree(feature: pd.Series, target: pd.Series) -> float:
        """
        Обучение решающего пня и вычисление корреляции между
        прогнозами, полученным с помощью решающего пня, и вектором
        целевой переменной.

        Parameters
        ----------
        feature: pandas.Series, shape = [n_samples, ]
            Вектор значений признака.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        score: float
            Значение корреляции.

        """
        feature = feature.fillna(-9999).values.reshape(-1, 1)
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(feature, target)

        prediction = tree.predict(feature)
        score = np.corrcoef(prediction, target)

        return np.round(100 * score[0, 1], 2)

    def fit(self, data, target=None):
        """
        Вычисление коэффициента корреляции для всех признаков.

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        self

        """
        if self.cat_features:
            missed_cols = list(
                set(self.cat_features) - set(data.columns)
            )
            if missed_cols:
                raise MissedColumnError(
                    f"Missed {list(missed_columns)} columns in data."
                )

            numeric_features = list(
                set(data.columns) - set(self.cat_features)
            )
        else:
            numeric_features = data.columns

        for feature in tqdm(numeric_features, leave=False):

            self.scores[feature] = self.calculate_tree(
                data[feature], target
            )

        return self

    def transform(self, data, target=None):
        """
        Отбор переменных по self.threshold.
        Если коффициент корреляции для переменной выше порога, 
        то переменная помечается 1 (использовать для дальнейшего анализа), 
        если ниже порога - маркируется 0 (не использовать для 
        дальнейшего анализа).

        Parameters
        ----------
        data: pandas.DataFrame, shape = [n_samples, n_features]
            Матрица признаков для обучения.

        target: pandas.Series, shape = [n_samples, ]
            Вектор целевой переменной.

        Returns
        -------
        scores: pandas.DataFrame, shape = [n_features, 3]
            Датафрейм с корреляционным-анализом переменных.

        """
        self.check_is_fitted
        scores = pd.Series(self.scores)
        scores = pd.DataFrame({"Variable": scores.index, "Correlation": scores.values})
        scores["Correlation_abs"] = np.abs(scores["Correlation"])
        scores["Selected"] = np.where(
            scores.Correlation_abs > self.threshold, 1, 0)
        scores = scores.sort_values(by="Correlation_abs", ascending=False)
        scores = scores.drop("Correlation_abs", axis=1)
        scores = scores.fillna(0)

        if self.cat_features:
            cat_features_scores = pd.DataFrame({
                "Variable": self.cat_features,
                "Correlation": "категориальный признак",
                "Selected": 1
            })
            scores = scores.append(cat_features_scores)

        mask = scores["Selected"] == 1
        self.used_features = scores.loc[mask, "Variable"].tolist()

        return scores.reset_index(drop=True)


class CategoricalFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Подготовка категориальных признаков: поиск и замена
    пропусков на фиксированное значение, применение
    LabelEncoder'a для перевода категорий в целочисленные
    векторы.

    Parameters:
    -----------
    config: dict
        Словарь с настройками запуска ядра.

    fill_value: string, optional, default = "NA"
        Значение для заполнения пропущенных элементов.

    copy: bool, optional, default = True
        Если True, то создается копия data. Если False,
        то все преобразования data производится inplace.

    Attributes:
    -----------
    _unique_values: Dict[string: list]
        Словарь уникальных значений признака, для которых
        был применен метод fit. Ключ словаря - название
        категориального признака, значение - список уникальных
        значений данного признака.

    encoders: Dict[string: LabelEncoder()]
        Словарь encoder'ов для каждого категориального признака.
        Ключ словаря - название категориального признака,
        значение - экземпляр LabelEncoder(), для которого
        был применен метод fit.

    cat_features: List[str]
        Словарь строк с названием категориальных переменных.

    """
    def __init__(self,
                 config: dict,
                 fill_value: str = "NA",
                 copy: bool = True) -> None:
        self.fill_value = fill_value
        self.config = config
        self.copy = copy

        self.encoders = {}
        self._unique_values = {}
        self.cat_features = None

    def _copy(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.copy() if self.copy else data

    @property
    def check_is_fitted(self):
        if not self.encoders:
            msg = ("This estimator is not fitted yet. Call 'fit' with "
                   "appropriate arguments before using this estimator.")
            raise NotFittedError(msg)
        return True

    def _prepare_data_dtypes(self, series: pd.Series) -> pd.Series:
        """
        Подготовка данных для передачи данных на вход encoder'a:
            - замена пропусков на fill_value;
            - преобразованеи столбца значений в object-столбец.

        Parameters:
        -----------
        series: pandas.Series
            Вектор наблюдений.

        Returns:
        --------
        series_prepared: pandas.Series
            Преобразованный вектор наблюдений.
        """
        try:
            if series.dtype == 'category':
                series = series.cat.add_categories(self.fill_value)
        except TypeError:
            pass
        series_prepared = series.fillna(self.fill_value)
        series_prepared = series_prepared.astype("str")
        return series_prepared

    def _find_new_values(self, series: pd.Series) -> pd.Series:
        """
        Поиск новых значений категориального признака, которые
        не были обработаны методом fit. Новые значения категории
        заменяются на fill_value, если fill_value был обработан
        методом fit, иначе - заменяются на первую обработанную
        категорию.

        Parameters:
        -----------
        series: pandas.Series
            Вектор наблюдений.

        Returns:
        --------
        series: pandas.Series
            Преобразованный вектор наблюдений.
        """
        observed_values = np.unique(series)
        expected_values = self._unique_values[series.name]
        new_values = list(set(observed_values) - set(expected_values))

        if new_values:
            bad_values_mask = (series.isin(new_values))
            series[bad_values_mask] = self.fill_value if self.fill_value in \
                expected_values else expected_values[0]

        return series

    def fit(self, data, target=None):
        """
        Обучение LabelEncoder'a

        Parameters:
        -----------
        data: pandas.DataFrame
            Матрица признаков.

        Returns:
        --------
        self: CategoricalFeaturesTransformer
        """
        if self.cat_features is None:
            self.cat_features = find_categorical_features(
                data, config=self.config)
            
        cat_features = find_categorical_features(
            data, config=self.config)
        
        new_features = sorted(list(set(cat_features) - set(self.cat_features)))
        
        if not new_features:
            new_features = self.cat_features
            
        for feature in tqdm(new_features, leave=False):
            if feature not in self.cat_features:
                self.cat_features.append(feature)
            x_prepared = self._prepare_data_dtypes(data[feature])
            self._unique_values[feature] = np.unique(x_prepared).tolist()
            self.encoders[feature] = LabelEncoder().fit(x_prepared)
        return self

    def transform(self, data, target=None):
        """
        Преобразование data, используя LabelEncoder.

        Parameters:
        -----------
        data: pandas.DataFrame
            Матрица признаков.

        Returns:
        --------
        data_transformed: pandas.DataFrame
            Преобразованная матрица признаков.
        """
        self.check_is_fitted
        x_transformed = self._copy(data)
        encoded_features = list(set(self.cat_features) & set(data.columns))

        for feature in tqdm(encoded_features, leave=False):
            x_grouped = self._prepare_data_dtypes(x_transformed[feature])
            x_grouped = self._find_new_values(x_grouped)

            encoder = self.encoders[feature]
            x_transformed[feature] = encoder.transform(x_grouped)

        return x_transformed


def encode_categorical(config, transformer = None, **eval_sets):
    """
    Применение CategoricalEncoder для каждого набора данных в eval_sets.

    Parameters
    ----------
    config: dict
        Конфигурационный файл.
        
    transformer: Optional[CategoricalFeaturesTransformer]
        Подготовленный заранее трансформер категориальных фич

    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Словарь с выборками, для которых требуется рассчитать статистику.
        Ключ словаря - название выборки (train / valid / ...), значение -
        кортеж с матрицей признаков (data) и вектором ответов (target).

    Returns
    -------
    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Преобразованный eval_sets.

    """
    if transformer is None:
        transformer = CategoricalFeaturesTransformer(config)
    
    train = transformer.fit_transform(eval_sets["train"][0])
    eval_sets["train"] = (train, eval_sets["train"][1])

    transformed_samples = [name for name in eval_sets if name != "train"]
    for sample in transformed_samples:
        df = transformer.transform(eval_sets[sample][0])
        eval_sets[sample] = (df, eval_sets[sample][1])

    return eval_sets, transformer
