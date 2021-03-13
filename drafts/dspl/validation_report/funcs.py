import pandas as pd
import numpy as np
import shap
import pickle
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Help functions
def _to_frame(X: pd.DataFrame, values: np.array, prefix: str) -> pd.DataFrame:
    """
    Функция для создания датафрейма с отранжированными значениями.

    Parameters:
    -----------
    X: pandas.DataFrame
        Матрица признаков.

    values: numpy.array
        Вектор с оценками важности признаков.

    prefix: string
        Префикс для колонки importance.

    Returns:
    --------
    df: pandas.DataFrame
        Датафрейм с отранжированными значениями.
    """
    df = pd.DataFrame({
        "feature": X.columns,
        f"{prefix}_importance": values
    })
    df = df.sort_values(by=f"{prefix}_importance", ascending=False)
    df = df.reset_index(drop=True)
    return df


def create_pred_df(model_info: tuple, X: pd.DataFrame, y: pd.Series):
    """
    Функция для построения pandas.DataFrame со значением целевой метки,
    прогнозами модели

    Parameters:
    -----------
    model_info: Tuple[sklearn.model, List[str]]
        Кортеж, первый элемент - обученный экземпляр модели,
        второй элемент - список используемых признаков модели.

    X: pandas.DataFrame
        Матрица признаков.

    y: pandas.Series
        Матрица целевой переменной.

    Returns:
    --------
    df: pandas.DataFrame
        Датафрейм с прогнозами.
    """
    estimator, features = model_info

    if getattr(estimator, "transform", None):
        y_pred = estimator.transform(X[features])

    elif getattr(estimator, "predict_proba", None):
        if str(type(estimator)).endswith("RandomForestClassifier'>"):
            y_pred = estimator.predict_proba(X[features]).fillna(-9999)[:, 1]
        else:
            y_pred = estimator.predict_proba(X[features])[:, 1]

    elif getattr(estimator, "predict", None):
        y_pred = estimator.predict(X[features])

    else:
        raise AttributeError("Estimator must have predict, predict_proba or "
                             "transform method")
    return pd.DataFrame({"y_true": y, "y_pred": y_pred})


def _validate_input_dataframe(data: pd.DataFrame):
    """
    Проверка входного датафрейса на валидность,
    датафрейм признается валидным, если содержит колонки
    y_pred, y_true. Если датасет не проходит проверку на валидность,
    будет выброшено MissedColumnError.

    Parameters:
    -----------
    data: pandas.DataFrame
        Матрица признаков.
    """
    required_cols = ["y_pred", "y_true"]
    missed_columns = list(
        set(required_cols) - set(data.columns)
    )

    if missed_columns:
        raise MissedColumnError(
            f"Missed {list(missed_columns)} columns.")


def calculate_permutation_feature_importance(estimator,
                                             metric,
                                             y: pd.Series,
                                             X: pd.DataFrame,
                                             fraction_sample: float = 0.15
                                             ) -> pd.DataFrame:
    """
    Функция для расчета важности переменных на основе перестановок.
    Подход к оценке важности признаков основан на изменении метрики
    при перемешивании значений данного признака. Если значение метрики
    уменьшается, значит признак важен для модели, если значение метрики
    увеличивается, то признак для модели не важен и его стоит исключить.

    Parameters:
    -----------
    estimator: sklearn.estimator
        Экземпляр модели, которая поддерживает API sklearn.
        Ожидается, что модель обучена, т.е. был вызван метод fit ранее.

    metric: func, sklearn.metrics
        Функция для оценки качества модели.

    X: pandas.DataFrame
        Матрица признаков.

    y: pandas.Series
        Вектор целевой переменной.

    fraction_sample: float, optional, default = 0.15
        Доля наблюдений от X для оценки важности признаков.

    Returns:
    --------
    X_transformed: pandas.DataFrame
        Преобразованная матрица признаков.
    """
    if fraction_sample > 1:
        raise ValueError(
            f"fraction_sample must be in range (0, 1], "
            f"but fraction_sample is {fraction_sample}")
    if isinstance(X, pd.DataFrame):
        x = X.copy()
        x, _, y, _ = train_test_split(
            x, y, train_size=fraction_sample, random_state=1)
    else:
        raise TypeError(
            f"x_valid must be pandas.core.DataFrame, "
            f"but x_valid is {type(X)}")

    feature_importance = np.zeros(x.shape[1])
    baseline_prediction = create_pred_df((estimator, x.columns),
                                         x, y)["y_pred"]
    baseline_score = metric(y, baseline_prediction)

    for num, feature in tqdm(enumerate(x.columns)):
        x[feature] = np.random.permutation(x[feature])
        score = metric(y, create_pred_df((estimator, x.columns),
                                         x, y)["y_pred"])
        feature_importance[num] = score
        x[feature] = X[feature]

    feature_importance = (baseline_score - feature_importance)\
        / baseline_score # * 100
    return _to_frame(x, feature_importance, "permutation")


def plot_permutation_importance(df: pd.DataFrame
                                , x_column: str
                                , name: str) -> None:
    """
    Построение графика важности фич по перестановкам.
    
    Parameters:
    -----------
    df: pandas.DataFrame
        Датафрейм со списком фич (features) и значением
        permutation importance (permutation_importance)
    x_column: str
        имя колонки, со значениями, которые откладывать по оси X
    name: string
        Имя файла для сохранения графика.
    Returns:
    -----------
    lines: 
        Список `.Line2D` объектов, графически представляющие данные.
    """
    plt.figure(figsize=(10, 6))
    plt.grid()
    n = len(df.columns) - 1
    color = iter(plt.cm.rainbow(np.linspace(0, 1, n)))
    leg = []
    for col in df.drop(x_column, axis=1).columns:
        c = next(color)
        plt.plot(df[x_column]
                 , df[col]
                 , c=c
                 , linewidth=3
                 , marker='o'
                 , markersize=12)
        leg.append(col)
    plt.legend(leg)
    plt.xticks(df[x_column], rotation="vertical")
    plt.xlabel(x_column)
    plt.ylabel("permutation_importance")
    plt.tight_layout()
    plt.savefig(f"{name}.png", bbox_inches="tight")


def plot_rec_curve(y_error: list, cumm_accur: list, y_error_baseline: list,
                   cumm_accur_baseline: list, name: str) -> None:
    """
    Построение графика REC curve для двух моделей

    Parameters:
    -----------
    y_error: list
        отсортированный список уровней ошибки на  модели

    cumm_accur:list
        куммулятивная доля наблюдений с заданным уровнем ошибки

    y_error_baseline: list
        отсортированный список уровней ошибки на baseline модели

    cumm_accur_baseline:
        куммулятивная доля наблюдений с заданным уровнем ошибки

    name: str
        наименование файла для сохрнения графика

    Returns:
    -----------
    lines:
        Список `.Line2D` объектов, графически представляющие данные.
    """

    plt.figure(figsize=(12, 8))
    plt.fill_between(y_error_baseline, cumm_accur_baseline, 1,
                     color="lightblue", label="AOC baseline prediction")
    plt.plot(y_error_baseline, cumm_accur_baseline, c="blue", linestyle="--",
             label="mean prediction")
    plt.fill_between(y_error, cumm_accur, 1, color="coral", label="AOC model")
    plt.plot(y_error, cumm_accur, c="crimson", linestyle="--", label="model")
    plt.grid()
    plt.legend(loc="lower right")
    plt.xlabel("absolute error", size=14)
    plt.ylabel("Accuracy, %", size=14)

    plt.savefig(f"{name}.png", bbox_inches="tight")


def plot_shap_importance(estimator,
                         df: pd.DataFrame,
                         features_list: list,
                         name: str) -> None:
    """
    Функция считает и рисует Shapley feature importance для обученной модели.
    Максимальное количество признаков для отображения на графике - 100

    Для модели Catboost на данный момент нет работающей реализации shap

    Parameters:
    -----------
    estimator: sklearn.estimator
        Экземпляр модели, которая поддерживает API sklearn.
        Ожидается, что модель обучена, т.е. был вызван метод fit ранее.

    df: pd.DataFrame
        Датафрейм с выборкой, на которой считать значимость

    features_list: list
        Список фичей модели, для которых считать значение важности

    name: str
        Путь к файлу для сохранения (включает имя файла без типа)

    """

    if str(type(estimator)).endswith("CatBoostClassifier'>"):
        print("Shap explainer does not support Catboost models yet")
    else:
        plt.figure()
        plt.title("Shapley feature impact plot")
        explainer = shap.TreeExplainer(estimator)

        sample = df[features_list].sample(n=20000, replace=False)
        shap_values = explainer.shap_values(sample)

        if str(type(estimator)).endswith("sklearn.LGBMClassifier'>"):
            shap.summary_plot(shap_values[1],
                              sample,
                              max_display=100,
                              show=False)
            
        elif str(type(estimator)).endswith("RandomForestClassifier'>"):
            shap.summary_plot(shap_values[1],
                              sample,
                              max_display=100,
                              show=False)
            
        else:
            shap.summary_plot(shap_values,
                              sample,
                              max_display=100,
                              show=False)
        plt.savefig(f"{name}.png", bbox_inches="tight")


def plot_shap_importance_summary(estimator,
                                 df: pd.DataFrame,
                                 features_list: list,
                                 name: str) -> None:
    """
    Функция считает и рисует Shapley feature importance summary
    для обученной модели

    Для модели Catboost на данный момент нет работающей реализации shap

    Parameters:
    -----------
    estimator: sklearn.estimator
        Экземпляр модели, которая поддерживает API sklearn.
        Ожидается, что модель обучена, т.е. был вызван метод fit ранее.

    df: pd.DataFrame
        Датафрейм с выборкой, на которой считать значимость

    features_list: list
        Список фичей модели, для которых считать значение важности

    name: str
        Путь к файлу для сохранения (включает имя файла без типа)

    """

    if str(type(estimator)).endswith("CatBoostClassifier'>"):
        print("Shap explainer does not support Catboost models yet")
    else:
        plt.figure()
        plt.title("Shapley feature impact plot")
        explainer = shap.TreeExplainer(estimator)

        sample = df[features_list].sample(n=20000, replace=False)
        shap_values = explainer.shap_values(sample)

        if str(type(estimator)).endswith("sklearn.LGBMClassifier'>"):
            shap.summary_plot(shap_values[1],
                              sample,
                              max_display=100,
                              show=False,
                              plot_type="bar")
            
        elif str(type(estimator)).endswith("RandomForestClassifier'>"):
            shap.summary_plot(shap_values[1],
                              sample,
                              max_display=100,
                              show=False,
                              plot_type="bar")
            
        else:
            shap.summary_plot(shap_values,
                              sample,
                              max_display=100,
                              show=False,
                              plot_type="bar")
    plt.savefig(f"{name}.png", bbox_inches="tight")


def prepare_datasets(config: dict) -> dict:
    """
    Функция подготовки датафреймов с выборками для формирования отчета
    """
    eval_sets = {}
    try:
        target = config["target_name"]
    except KeyError:
        print("Target variable is not defined!")

    for ds_name in ["train", "test", "valid", "OOT"]:
        path_name = f"{ds_name}_path"
        data_path = config.get(path_name, None)
        login = config.get("login", None)
        password = config.get("password", None)
        if data_path is not None:

            data = read_data(data_path, login, password)
            eval_sets[ds_name] = (data.drop(target, axis=1),
                                    data[target])

    return eval_sets


def prepare_model(config: dict) -> dict:
    """
    подготовка модели для формирования отчета
    """

    model_path = config.get("model_path", None)
    model_name = ".".join(model_path.split("/")[-1].split(".")[:-1])

    cat_features_path = config.get("categorical_features_path", None)

    if cat_features_path is not None:
        # Читать список категориальных фичей
        with open(cat_features_path,"rb") as cat_feat:
            cat_features = pickle.load(cat_feat)
        # заполнить конфигуарционный файл списком кат. фичей
        config["categorical_features"] = cat_features
        
    # Читать модель
    with open(model_path, "rb") as mdl:
        model = pickle.load(mdl)

    features_path = config.get("model_features", None)
    with open(features_path, "rb") as feat:
        features_list = pickle.load(feat)

    return {model_name: (model, features_list)}


def get_estimator_params(estimator) -> pd.DataFrame():
    """
    Достает список и значения параметров из модели
    Сохраняет в pandas dataframe
    """
    
    try: 
        params = estimator.params
    except AttributeError:
        params = estimator.get_params()

    params_df = pd.DataFrame(data=params.values(),
                             columns=["Value"],
                             index=list(params.keys()))
    params_df.index.name = "Parameter"
    params_df.reset_index(inplace=True)

    return params_df
