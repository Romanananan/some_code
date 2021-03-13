import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Union, Optional
from sklearn.model_selection import train_test_split


def _to_frame(data: pd.DataFrame, values: np.array, prefix: str) -> pd.DataFrame:
    """
    Функция для создания датафрейма с отранжированными значениями.

    Parameters
    ----------
    data: pandas.DataFrame
        Матрица признаков.

    values: numpy.array
        Вектор с оценками важности признаков.

    prefix: string
        Префикс для колонки importance.

    Returns
    -------
    data: pandas.DataFrame
        Датафрейм с отранжированными значениями.
    """
    data = pd.DataFrame({
        "feature": data.columns,
        f"{prefix}_importance": values
    })
    data = data.sort_values(by=f"{prefix}_importance", ascending=False)
    data = data.reset_index(drop=True)
    return data


def calculate_permutation_feature_importance(estimator,
                                             metric: callable,
                                             data: pd.DataFrame,
                                             target: pd.Series,
                                             fraction_sample: float = 0.15,
                                             maximize: bool = True
                                             ) -> pd.DataFrame:
    """
    Функция для расчета важности переменных на основе перестановок.
    Подход к оценке важности признаков основан на изменении метрики
    при перемешивании значений данного признака. Если значение метрики
    уменьшается, значит признак важен для модели, если значение метрики
    увеличивается, то признак для модели не важен и его стоит исключить.

    Parameters
    ----------
    estimator: sklearn.estimator
        Экземпляр модели, которая поддерживает API sklearn.
        Ожидается, что модель обучена, т.е. был вызван метод fit ранее.

    metric: func, sklearn.metrics
        Функция для оценки качества модели.

    data: pandas.DataFrame
        Матрица признаков.

    target: pandas.Series
        Вектор целевой переменной.

    fraction_sample: float, optional, default = 0.15
        Доля наблюдений от data для оценки важности признаков.

    maximize: boolean, optional, default = True
        Флаг максимизации метрики. Означает, что чем выше значение метрики,
        тем качественее модель. Опциональный параметр, по умолчанию, равен True.

    Returns:
    --------
    data_transformed: pandas.DataFrame
        Преобразованная матрица признаков.

    """
    if fraction_sample > 1:
        raise ValueError(
            f"fraction_sample must be in range (0, 1], "
            f"but fraction_sample is {fraction_sample}")
    if isinstance(data, pd.DataFrame):
        data_ = data.copy()
        x, _, y, _ = train_test_split(
            data_, target, train_size=fraction_sample, random_state=1)
    else:
        raise TypeError(
            f"x_valid must be pandas.core.DataFrame, "
            f"but x_valid is {type(data)}")

    feature_importance = np.zeros(x.shape[1])
    try:
        baseline_prediction = estimator.transform(x)
    except AttributeError:
        baseline_prediction = estimator.predict_proba(x)[:, 1]
    baseline_score = metric(y, baseline_prediction)

    for num, feature in tqdm(enumerate(x.columns), leave=False):
        x[feature] = np.random.permutation(x[feature])
        try:
            score = metric(y, estimator.transform(x))
        except AttributeError:
            score = metric(y, estimator.predict_proba(x)[:, 1])
        feature_importance[num] = score
        x[feature] = data_[feature]

    if maximize:
        feature_importance = (baseline_score - feature_importance) * 100
    else:
        feature_importance = (feature_importance - baseline_score)
    return _to_frame(x, feature_importance, "permutation")


def select_subset_features(features_imp: pd.DataFrame,
                           threshold_value: Union[float, int] = None,
                           top_k: Optional[int] = None) -> pd.DataFrame:
    """
    Функция для отбора признаков.
    Отбор признаков осуществляется на основе меры важностей признаков.
    Если задан threshold_value - то отбор производится по порогу меры
    важности (остаются признаки, мера важности которых выше заданного
    порога), если задан top_k - то отбираются top-k признаков по
    заданной мере важности.

    Parameters:
    -----------
    features_imp: pandas.DataFrame, shape = [n_features, 2]
        Матрица с оценкой важности признаков.

    threshold_value: float / int, optional, default = None.
        Пороговое значение меры важности признаков.

    top_k: int, optional, default = None.
        Максимальное количество признаков.

    Returns:
    --------
    used_features: list
        Список используемых признаков.

    """
    col_name = [col for col in features_imp.columns if "importance" in col][0]
    if threshold_value:
        valid_features = features_imp[features_imp[col_name] > threshold_value]
        return valid_features["feature"].tolist()
    if top_k:
        valid_features = features_imp.head(n=top_k)
        return valid_features["feature"].tolist()
    else:
        message = ("Incorrect params. Set the params: threshold_value / top_k."
                   f"Current params: threshold_value = {threshold_value}, "
                   f"top_k = {top_k}.")
        raise ValueError(message)


def find_categorical_features(data: pd.DataFrame, config: dict) -> np.array:
    """
    Функция поиска категориальных переменных в датасете.
    Поиск осуществляется по типам: к категориальным признакам
    относятся колонки типа object / category, кроме колонок,
    которые указанные как drop_features.

    Parameters
    ----------
    data: pandas.DataFrame
        Матрица признаков.

    config: dict, optional, default = config_file
        Словарь с конфигурацией запуска кернела.

    Returns
    -------
    cat_features: numpy.array
        Список категориальных признаков.

    """
    object_cols = data.dtypes[data.dtypes == "object"].index
    category_cols = data.dtypes[data.dtypes == "category"].index
    cat_features = object_cols.append(category_cols).tolist()
    config_categories = config.get("categorical_features", [])

    if config_categories:
        for cat_feature in config_categories:
            if cat_feature in data.columns:
                cat_features.append(cat_feature)
    
    non_cat_features = config.get('non_categorical_features')
    if non_cat_features is not None:
        cat_features = sorted(list(set(cat_features) - set(non_cat_features)))
    
    return np.unique(cat_features).tolist()


def drop_features(config, **eval_sets):
    """
    Удаление признаков из каждого набора данных в eval_sets.
    Удаляются признаки, которые размещены в конфигурационном
    файла с ключам drop_features и target_name.

    Parameters
    ----------
    config: dict
        Конфигурационный файл.

    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Словарь с выборками, для которых требуется рассчитать статистику.
        Ключ словаря - название выборки (train / valid / ...), значение -
        кортеж с матрицей признаков (data) и вектором ответов (target).

    Returns
    -------
    eval_sets: Dict[string, Tuple[pandas.DataFrame, pandas.Series]]
        Преобразованный eval_sets.

    """
    garbage_features = config.get("drop_features", [])
    garbage_features.append(config.get("target_name"))
    for sample in eval_sets:
        data, target = eval_sets[sample]
        garbage_features_intersection = list(set(garbage_features).intersection(set(data.columns)))
        data = data.drop(garbage_features_intersection, axis=1)
        eval_sets[sample] = (data, target)

    return eval_sets
