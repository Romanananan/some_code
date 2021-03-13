import numpy as np
import pandas as pd


def filter_outliers(data: pd.DataFrame, target: pd.Series, config: dict):
    """
    Вычисление и удаление выбросов из данных.

    Parameters
    ----------
    data: pandas.DataFrame, shape = [n_samples, n_features]
        Матрица признаков.

    target: pandas.Series, shape = [n_samples, ]
        Вектор целевой переменной.

    config: dict
        Конфигурационный файл эксперимента.

    Returns
    -------
    transformed_data: Tuple[pd.DataFrame, pd.Series]
        Кортеж, где первый элемент - преобразованная матрица
        признаков, второй элемент - преобразованный вектор
        целевой переменной.

    """
    min_perc = np.percentile(target, q=config.get("min_percentile", 1))
    max_perc = np.percentile(target, q=config.get("max_percentile", 99))
    target_mask = (target > min_perc) & (target < max_perc)

    data = data.loc[target_mask]
    data = data.reset_index(drop=True)

    target = target.loc[target_mask]
    target = target.reset_index(drop=True)

    return data, target


def filter_outliers_in_eval_set(eval_sets, config):
    """
    Удаление выбросов в eval_set. Выбросы удаляются в
    выборках с ключом train и valid. Если в eval_set
    присутствует выборка с ключом test, то создается
    дополнительная выборка с ключом test_without_outliers,
    где выбросы удаляются, в выборке с ключом test выбросы
    не удаляются. Если в eval_set присутствует выборка с
    ключом OOT, то в ней выбросы не удаляются.

    Parameters
    ----------
    eval_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Словарь, где ключ - название выборки, значение -
        кортеж с обучающей выборкой и вектором целевой
        переменной.

    config: dict
        Конфигурационный файл эксперимента.

    Returns
    -------
    eval_sets: Dict[str, Tuple[pd.DataFrame, pd.Series]]
        Преобразованный словарь, где ключ - название
        выборки, значение - кортеж с обучающей выборкой и
        вектором целевой переменной.

    """
    eval_sets["train"] = filter_outliers(*eval_sets["train"], config)
    eval_sets["valid"] = filter_outliers(*eval_sets["valid"], config)

    if "test" in eval_sets:
        eval_sets["test2"] = filter_outliers(
            *eval_sets["test"], config)
    return eval_sets
