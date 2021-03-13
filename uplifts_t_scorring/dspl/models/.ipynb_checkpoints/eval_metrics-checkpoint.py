from typing import Tuple
import numpy as np
import xgboost as xgb


def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    RMSLE для XGBoost
    """
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return "rmsle", float(np.sqrt(np.sum(elements) / len(y)))


def mse(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    MSE для XGBoost
    """
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(y - predt, 2)
    return "mse", float(np.sqrt(np.sum(elements) / len(y)))


def r2(predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
    """
    R2 для XGBoost
    """
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(y - predt, 2)
    mse = np.sqrt(np.sum(elements) / len(y))
    return "r2", float(1 - mse / np.var(y))