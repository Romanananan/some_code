import numpy as np
import pandas as pd

from tqdm import tqdm


def compress_object_types(data: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасное сжатие object-типов.

    Parameters
    ----------
    data: pd.DataFrame
        Матрица признаков для сжатия.

    Returns
    -------
    compressed_data: pd.DataFrame
        Матрциа признаков со сжатыми int-признаками.

    """
    feature_mask = data.dtypes == "object"
    category_mask = data.loc[:, feature_mask].nunique(axis=0) / len(data) < .5
    data.loc[:, feature_mask & category_mask] = data.loc[:, feature_mask & category_mask].astype('category')
    
    return data

def compress_int_type(data: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасное сжатие int-типов (int64 -> int32).

    Parameters
    ----------
    data: pd.DataFrame
        Матрица признаков для сжатия.

    Returns
    -------
    compressed_data: pd.DataFrame
        Матрциа признаков со сжатыми int-признаками.

    """
    mask = data.dtypes == "int64"
    features = data.dtypes[mask].index
    data[features] = data[features].astype(np.int32)

    return data

def compress_float_type(data: pd.DataFrame) -> pd.DataFrame:
    """
    Безопасное сжатие int-типов (float64 -> float32).

    Parameters
    ----------
    data: pd.DataFrame
        Матрица признаков для сжатия.

    Returns
    -------
    compressed_data: pd.DataFrame
        Матрица признаков со сжатыми float-признаками.


    """
    mask = data.dtypes == "float64"
    features = data.dtypes[mask].index
    data[features] = data[features].astype(np.float32)

    return data

def safe_compress_data(data: pd.DataFrame,
                       copy: bool = True,
                       deep: bool = True) -> pd.DataFrame:
    """
    Безопасное сжатие данных в pandas.DataFrame.

    Parameters
    -----------
    data: pandas.DataFrame
        Матрица признаков.

    copy: boolean, optional, default = True
        Признак использования копии датасета.
        Опциональный параметр, по умолчанию True.

    deep: boolean, optional, default = True
        Флаг точного расчета занимаемой памяти. Если значение
        True - то производится расчет точного объема потребляемой
        памяти, если значение False - то ориентировочно. Точный
        расчет производится дольше.

    Returns
    -------
    compressed_data: pandas.DataFrame
        Сжатая матрица признаков.

    """
    data = data.copy() if copy else data
    start_memory_usage = data.memory_usage(deep=deep).sum() / 1024**2
    print(f"data.memory_usage: {round(start_memory_usage, 2)} Mb")

    data = compress_object_types(data)
    data = compress_float_type(data)
    data = compress_int_type(data)

    end_memory_usage = data.memory_usage(deep=deep).sum() / 1024**2
    reduced = 100 * (start_memory_usage - end_memory_usage) / start_memory_usage
    print(f"data.memory_usage: {round(end_memory_usage, 2)} Mb")
    print(f"Reduced: {round(reduced, 2)}%")

    return data