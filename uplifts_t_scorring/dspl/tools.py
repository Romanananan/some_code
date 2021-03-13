# Evdokimov Nikolay, Sberbank

import logging
import multiprocessing as mp
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import time
from collections import deque
from functools import partial
from inspect import cleandoc
from itertools import chain, starmap
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from pprint import pprint
from sys import getsizeof, stderr
import warnings

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import teradata
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import lightgbm

try:
    import paramiko
except ModuleNotFoundError:
    pass

n_cores = mp.cpu_count()


def isnotebook():
    """Am I in jupyter?

    Returns
    -------
    bool

    References
    ----------
    Based on https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


# if isnotebook():
#    from tqdm import tqdm_notebook as tqdm
# else:


def psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """Calculate the PSI (population stability index) across all variables

    Parameters
    ----------
    expected: numpy.2darray
        numpy matrix of original values
    actual: numpy.2darray
        numpy matrix of new values, same size as expected
    buckettype: str, optional
        type of strategy for creating buckets, bins splits into even splits
        , quantiles splits into quantile buckets
    buckets: int, optional
        number of quantiles to use in bucketing variables
    axis: int, optional
        axis along which variables' values are distributed, 0 for vertical, 1 for horizontal

    Returns
    -------
    psi_values: numpy.1darray
        psi values for each variable

    References
    ----------
    Based on https://github.com/mwburke/population-stability-index
    """

    def psi_(expected_array, actual_array, buckets_):
        """Calculate the PSI for a single variable

        Parameters
        ----------
        expected_array: numpy.1darray
            array of original values
        actual_array: numpy.1darray
            array of new values, same size as expected
        buckets_: int
            number of percentile ranges to bucket the values into

        Returns
        -------
        psi_value: float64
            calculated PSI value
        """

        def scale_range(input_, min_, max_):
            input_ += -(np.min(input_))
            input_ /= np.max(input_) / (max_ - min_)
            input_ += min_
            return input_

        breakpoints = np.arange(0, buckets_ + 1) / buckets_ * 100

        if buckettype == 'bins':
            breakpoints = scale_range(breakpoints, np.min(
                expected_array), np.max(expected_array))
        elif buckettype == 'quantiles':
            breakpoints = np.stack(
                [np.percentile(expected_array, b) for b in breakpoints])

        expected_percents = np.histogram(expected_array, breakpoints)[
            0] / len(expected_array)
        actual_percents = np.histogram(actual_array, breakpoints)[
            0] / len(actual_array)

        def sub_psi(e_perc, a_perc):
            """Calculate the actual PSI value from comparing the values.
               Update the actual value to a very small number if equal to zero
            """
            if a_perc == 0:
                a_perc = 0.0001
            if e_perc == 0:
                e_perc = 0.0001

            value = (e_perc - a_perc) * np.log(e_perc / a_perc)
            return value

        psi_value = np.sum(sub_psi(expected_percents[j], actual_percents[j]) for j in range(
            0, len(expected_percents)))

        return psi_value

    psi_values = np.empty(1 if len(expected.shape) ==
                          1 else expected.shape[axis])

    for i in range(0, len(psi_values)):
        if len(psi_values) == 1:
            psi_values = psi_(expected, actual, buckets)
        elif axis == 0:
            psi_values[i] = psi_(expected[:, i], actual[:, i], buckets)
        elif axis == 1:
            psi_values[i] = psi_(expected[i, :], actual[i, :], buckets)

    return psi_values


def replace_str_dict(string: str, mappings: dict) -> str:
    """Replace parts of a string based on a dictionary.

    Parameters
    ----------
    string: str
        The string to replace characters in.
    mappings: dict
        A dictionary of replacement mappings.

    Returns
    -------
    replaced_string: str

    Examples
    --------
    This function takes a string a dictionary of
    replacement mappings. For example, if I supplied
    the string "Hello world.", and the mappings
    {"H": "J", ".": "!"}, it would return "Jello world!".
    """
    replaced_string = string
    for character, replacement in mappings.items():
        replaced_string = replaced_string.replace(character, replacement)
    return replaced_string


def parallel_wrapper(func, path, *args, **kwargs):
    """Wrapper to run joblib parallel tasks and save results in file

    Parameters
    ----------
    func: object
        function name, you need to parallelize
    path: str
        filename path, to save results to
    args:
        positional arguments with iterable objects to slice and pass to func
    kwargs:
        optional func arguments
    Returns
    -------
    None
    """
    if kwargs:
        mapfunc = partial(func, **kwargs)
    else:
        mapfunc = func
    with open(path, 'wb') as file:
        pickle.dump(Parallel(n_jobs=n_cores)(
            starmap(delayed(mapfunc), tqdm(zip(*args), total=len(args[0])))), file)


class isolate:
    """Wrapper to run memory isolated functions

    Examples
    --------
    Simple usage:
    >>> def dummy(a, b, keyword_argument=None):
    >>>     print (a, b, keyword_argument)
    >>> with isolate(dummy) as isolated:
    >>>     isolated.run([1,2,3], [4,5,6], keyword_argument=2)
    [1, 2, 3] [4, 5, 6] 2

    More complex example:
    >>> def dummy(a, b, keyword_argument=None):
    >>>     return (a, b, keyword_argument)
    >>> with isolate(parallel_wrapper) as isolated:
    >>>     isolated.run(dummy, 'file.pkl', [1,2,3], [4,5,6], keyword_argument='ff')
    >>> with open('file.pkl', 'rb') as file:
    >>>     tmp = pickle.load(file)
    >>> tmp
    100%|██████████| 3/3 [00:00<00:00,  4.66it/s]
    [(1, 4, 'ff'), (2, 5, 'ff'), (3, 6, 'ff')]
    """

    def __init__(self, func):
        self.func = func
        self.p = None

    def __enter__(self):
        return self

    def run(self, func, *args, **kwargs):
        """Execute isolated function

        Parameters
        ----------
        func: object
            function name you want to isolate
        args:
            positional func arguments
        kwargs:
            optional func arguments

        Returns
        -------
        None
        """
        self.p = mp.Process(target=self.func, args=(
            func, *args), kwargs=kwargs)
        self.p.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.p.join()


def bytes_to_human(num, suffix='B'):
    """Format numeric bytes value into human-readable string

    Parameters
    ----------
    num:
        input numeric value
    suffix:
        string to append to format
    Returns
    -------
    str
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def mem_pie(
        o=None, handlers=None, tr=0.1, verbose=False):
    """Returns the approximate memory size of object or all objects in kernel

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

    Parameters
    ----------
    o: object, optional
        oject to get sizeof, otherwise of all objects in kernel
    handlers: dict, optional
        dictionary of custom handlers, which help to iterate over objects'
        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}
    tr: float, optional
        objects, that are smaller than tr*100% are grouped together
    verbose: bool, optional
        print object's params
    Returns
    -------
    None

    References
    ----------
    Based on http://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/

    Examples
    --------
    >>> import numpy as np
    >>> tmp1 = {'tmp1':np.random.rand(10000,10000)}
    >>> tmp2 = np.random.rand(10000,10000)
    >>> mem_pie()
    """
    if o is None:
        o = {
            name: value
            for name, value in globals().items()
            if not (
                    hasattr(name, '__call__')
                    or name.startswith('_')
                    or name.endswith('_')
                    or name in ['In', 'Out', 'get_ipython', 'exit', 'quit']
            )
        }
    if handlers is None:
        handlers = {}

    def dict_handler(d):
        return chain.from_iterable(d.items())

    def numpy_handler(n):
        return [n.nbytes]

    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                    np.ndarray: numpy_handler
                    }
    all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(obj):
        if id(obj) in seen:  # do not double count the same object
            return 0
        seen.add(id(obj))
        s = getsizeof(obj, default_size)

        if verbose:
            print(s, type(obj), repr(obj), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(obj, typ):
                s += sum(map(sizeof, handler(obj)))
                break
        return s

    if isinstance(o, dict):
        mem_df = pd.DataFrame(
            list({name: sizeof(value) for name, value in o.items()}.items()), columns=['object', 'size'])
        tr = mem_df['size'].sum() * tr
        mem_df.loc[mem_df['size'] < tr, 'object'] = '...'
        mem_df = mem_df.groupby('object').sum().sort_values('size', ascending=False)
        # Set figure size
        plt.figure(figsize=(10, 10))
        # Plot main pie
        patches, texts, _ = plt.pie(
            mem_df['size'].values, labels=mem_df.index, explode=np.repeat(
                0.01, mem_df['size'].shape[0])  # '%1.1f%%'
            , autopct=lambda pct: bytes_to_human(pct * mem_df['size'].sum() / 100), startangle=90
        )
        plt.legend(patches, mem_df.index)
        # Add dummy circle
        white_circle = plt.Circle((0, 0), 0.8, color='white')
        p = plt.gcf()
        p.gca().add_artist(white_circle)
        # Add total mem size
        plt.text(0, 0, bytes_to_human(mem_df['size'].sum()))
        # Show final plot
        plt.show()
    else:
        return sizeof(o)


def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error,
    excluding y_true==0 rows

    Parameters
    ----------
    y_true: numpy.1darray
        actual values
    y_pred: numpy.1darray
        predicted values

    Returns
    -------
    float
    """
    import warnings
    warnings.simplefilter('ignore')
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    result = abs(y_true - y_pred) / y_true
    result = result[~(np.isnan(result) | np.isinf(result))]
    return np.mean(result) * 100


def hparams_plot(trials, hp_list=None, blur=True, exclude_loss=None):
    """Plot hyper parameters' coloured scatter matrix

    Parameters
    ----------
    trials:
        hyperopt trials object
    hp_list: list of str
        parameters' names to plot
    blur: bool, optinal
        configures apperance of overlapping points
    exclude_loss: list of numerics
        exclude particular loss values from plotting

    Returns
    -------
    None
    """
    if hp_list is None:
        hp_list = []
    if exclude_loss is None:
        exclude_loss = []

    # We chose just a few params to inspect not to bloat the scatterplot:
    params_values = [
        [combination["params"][p]
            for combination in trials.results if combination["loss"] not in exclude_loss]
        for p in hp_list]
    best_accs = [combination["loss"]
                 for combination in trials.results if combination["loss"] not in exclude_loss]
    """Scatterplot colored according to the Z values of the points."""
    nb_params = len(params_values)
    best_accs = np.array(best_accs)
    norm = matplotlib.colors.Normalize(
        vmin=best_accs.min(), vmax=best_accs.max())

    # , facecolor=bg_color, edgecolor=fg_color)
    fig, ax = plt.subplots(nb_params, nb_params, figsize=(16, 16))

    for i in range(nb_params):
        p1 = params_values[i]
        for j in range(nb_params):
            if j <= i:
                p2 = params_values[j]
                axes = ax[i, j]
                # Subplot:
                if blur:
                    s = axes.scatter(p2, p1, s=400, alpha=.1,
                                     c=best_accs, cmap='viridis', norm=norm)
                    s = axes.scatter(p2, p1, s=200, alpha=.2,
                                     c=best_accs, cmap='viridis', norm=norm)
                    s = axes.scatter(p2, p1, s=100, alpha=.3,
                                     c=best_accs, cmap='viridis', norm=norm)
                s = axes.scatter(p2, p1, s=15, c=best_accs,
                                 cmap='viridis', norm=norm)
                # Labels only on side subplots, for x and y:
                if j == 0:
                    axes.set_ylabel(hp_list[i], rotation=90)
                else:
                    axes.set_yticks([])
                if i == nb_params - 1:
                    axes.set_xlabel(hp_list[j], rotation=0)
                else:
                    axes.set_xticks([])
            else:
                fig.delaxes(ax[i, j])

    fig.subplots_adjust(right=0.82, top=0.95)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(s, cax=cbar_ax)

    plt.suptitle(
        f"Scatterplot matrix of [{len(best_accs)}] hparams' combinations in the search space, colored in function of accuracy")
    plt.show()


class COMPRESS:
    """Сonverts numeric-like object columns into numeric ones and compresses types for all numeric columns
    """

    def __init__(self):
        self.fitted = False
        self.nulls_columns = None
        self.stats = None
        self.types = None

    def __repr__(self):
        return f'{self.__class__.__name__}: Fitted - {self.fitted!r} ({self.__dict__.keys()!r})'

    def fit(self, df, replace_nulls_inf=False):
        """Analyse data

        Parameters
        ----------
        df: pandas.DataFrame
            data to analyse
        replace_nulls_inf: bool, optional
            whether to fill NaN\inf with zeros before compression

        Returns
        -------
        self

        References
        ----------
        Based on https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        """
        # Convert to numeric if possible
        for var in list(df.dtypes[df.dtypes == object].index):    
            df[var] = pd.to_numeric(df[var], errors='ignore')
        # Get list of numerical columns
        numericals = list(df.dtypes[df.dtypes != object].index)
        # Get columns with nulls
        nulls = df[numericals].isnull()
        nulls_columns = nulls.any()
        self.nulls_columns = nulls_columns[nulls_columns == True].index.tolist(
        )
        if replace_nulls_inf:
            pd.set_option('use_inf_as_na', True)
            df = df.fillna(0)
            pd.set_option('use_inf_as_na', False)
            numericals_not_null = numericals
        else:
            numericals_not_null = list(set(numericals) - set(self.nulls_columns))
        # Calc min-max
        self.stats = (
            pd.DataFrame(
                np.vstack([
                    np.min(df[numericals].values, axis=0)
                    , np.max(df[numericals].values, axis=0)])
                .T
                , columns=['min', 'max']
                , index=numericals)
            .sort_index()
        )
        # Check if columns can be converted to an integer
        self.stats = self.stats.join(
            pd.Series(
                abs(
                    df[numericals_not_null].values
                    - df[numericals_not_null].values.astype(np.int64)
                )
                .sum(axis=0) 
                < 0.01
                , index = numericals_not_null
                , name = 'integerable'
            )
            .sort_index()
        ).fillna(False)
        # TODO: check datetime
        # Form astype mask
        self.types = pd.DataFrame(index=self.stats.index)
        self.types['type'] = None

        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= 0)
            & (self.stats['max'] <= np.iinfo(np.uint8).max), 'type'] = [np.uint8]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= 0)
            & (self.stats['max'] <= np.iinfo(np.uint16).max), 'type'] = [np.uint16]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= 0)
            & (self.stats['max'] <= np.iinfo(np.uint32).max), 'type'] = [np.uint32]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= 0)
            & (self.stats['max'] > np.iinfo(np.uint32).max), 'type'] = [np.uint64]

        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= np.iinfo(np.int8).min)
            & (self.stats['max'] <= np.iinfo(np.int8).max), 'type'] = [np.int8]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= np.iinfo(np.int16).min)
            & (self.stats['max'] <= np.iinfo(np.int16).max), 'type'] = [np.int16]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= np.iinfo(np.int32).min)
            & (self.stats['max'] <= np.iinfo(np.int32).max), 'type'] = [np.int32]
        self.types.loc[
            self.types['type'].isnull()
            & self.stats.integerable
            & (self.stats['min'] >= np.iinfo(np.int64).min)
            & (self.stats['max'] <= np.iinfo(np.int64).max), 'type'] = [np.int64]

        self.types.loc[
            ~self.stats.integerable, 'type'] = [np.float32]

        self.types = self.types.type.to_dict()
        self.fitted = True

        return self

    def transform(self, df, separate_null=False, replace_nulls_inf=False, drop_single=False):
        """Compress data

        Parameters
        ----------
        df: pandas.DataFrame
            Data to compress
        separate_null: bool, optional
            whether to create new flag column to indicate NaN value per all columns
        replace_nulls_inf: bool, optional
            whether to fill NaN\inf with zeros before compression
        drop_single: bool, optional
            whether to drop columns with single value, counts NaN as separate level

        Returns
        -------
        compressed pandas.DataFrame
        """
        if self.fitted:
            inner = df.copy()
            nulls_columns = [var for var in self.nulls_columns if var in inner.columns]
            types = {key:value for key, value in self.types.items() if key in inner.columns}
            # Push nulls to separate columns
            if separate_null:
                for var in nulls_columns:
                    varnull = var + '_null_flag'
                    inner[varnull] = inner[var].isnull()
            # Replace numerical nulls by zeros
            if replace_nulls_inf:
                pd.set_option('use_inf_as_na', True)
                inner[nulls_columns] = inner[nulls_columns].fillna(0)
                pd.set_option('use_inf_as_na', False)
            # Convert types
            inner = inner.astype(types)
            # Drop variables with single value
            if drop_single:
                to_drop_list = []
                for var in inner.columns:
                    if inner[var].nunique(dropna=False) == 1:
                        to_drop_list += [var, ]
                if to_drop_list:
                    inner = inner.drop(to_drop_list, axis=1)

            return inner
        else:
            raise AttributeError(
                'Сначала вызовите метод fit(data: pd.DataFrame)')

    def fit_transform(self, df, separate_null=False, replace_nulls_inf=True, drop_single=True):
        """Analyse and compress data

        Parameters
        ----------
        df: pandas.DataFrame
            Data to compress
        separate_null: bool, optional
            whether to create new flag column to indicate NaN value per all columns
        replace_nulls_inf: bool, optional
            whether to fill NaN\inf with zeros before compression
        drop_single: bool, optional
            whether to drop columns with single value, counts NaN as separate level

        Returns
        -------
        compressed pandas.DataFrame
        """
        return self.fit(df, replace_nulls_inf=replace_nulls_inf).transform(df, separate_null=separate_null, replace_nulls_inf=replace_nulls_inf, drop_single=drop_single)


def spawn_logger(
        logger_name, log_file='data/tmp.log', formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")):
    """Create logger object

    Parameters
    ----------
    logger_name: str
        name to print in log_file
    log_file: str
        path to save log file
    formatter: logging.Formatter
        format to create log lines

    Returns
    -------
    logger object

    References
    ----------
    Based on https://www.toptal.com/python/in-depth-python-logging
    """

    def get_console_handler(formatter_):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter_)
        return console_handler

    def get_file_handler(log_file_):
        file_handler = TimedRotatingFileHandler(log_file_, when='midnight')
        file_handler.setFormatter(formatter)
        return file_handler

    logger = logging.getLogger(logger_name)
    # better to have too much log than not enough
    logger.setLevel(logging.DEBUG)
    logger.addHandler(get_console_handler(formatter))
    logger.addHandler(get_file_handler(log_file))
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = False
    return logger


def td_download(
        obj=None, query="", fast=True, delimiter='^', dtype=None, decimal=',', debug=False, schema="tdsb15.cgs.sbrf.ru"
        , username="", password="", show_progress=True, compress=True, compress_separate_null=False, compress_replace_nulls_inf=True
        , compress_drop_single=True):
    """Download table or view from Teradata and load it to pandas.DataFrame

    Parameters
    ----------
    obj: str, optional
        'database.object' if 'query' is not specified
    query: str, optional
        select query if 'obj' is not specified
    fast: bool, optional
        use Teradata Parallel Transport tool to download data in parallel
    delimiter: str, optional
        delimiter to use in parsing data if 'fast' is specified
    dtype: dictionary, optional
        define column types for pandas while reading from buffer file
        , must be numerical istead of names
    decimal: str, optional
        delimiter for decimal numbers
    debug: bool, optional
        whether to delete temporary files
    schema: str, optional
        Teradata schema
    username: str
    password: str
    show_progress: bool, optional
        show progress info
    compress: bool, optional
        whether to convert numeric-like object columns into numeric ones and compress types for all numeric columns
    compress_separate_null: bool, optional
        whether to create new flag column to indicate NaN value per all columns
    compress_replace_nulls_inf: bool, optional
        whether to fill NaN\inf with zeros before compression
    compress_drop_single: bool, optional
        whether to drop columns with single value, counts NaN as separate level

    Returns
    -------
    pandas.DataFrame with header if 'obj' is specified, otherwise without header
    """
    if '' in [username, password]:
        raise Exception('Empty login or password.')
    if not obj and not query:
        raise Exception('You should specify either obj or query.')
    if fast:
        if obj:
            query = f"select * from {obj}"
        # Format query
        query = query.replace('\n', ' ').replace("'", "''")
        #
        local_seed = round(os.getpid()+time.time())
        path_tmp = os.path.expanduser(
            '~') + '/_' + str(local_seed)  # str(random.randint(0, 1000000))
        if os.path.exists(path_tmp):
            shutil.rmtree(path_tmp)
            os.mkdir(path_tmp)
        else:
            os.mkdir(path_tmp)
        # Create utility files
        txt = cleandoc("""
            SourceTdpId = '%s'
            , SourceUserName = '%s' 
            , SourceUserPassword = '%s'
            , DDLPrivateLogName = 'ddlprivate.log'
            , ExportPrivateLogName = 'exportprivate.log'
            , TargetErrorList = ['3807']
            , TargetFileName = '%s'
            , TargetFormat = 'delimited'
            , TargetTextDelimiter = '%s'
            , TargetOpenMode = 'write'
            , SelectStmt = '%s' """) % (schema, username, password, path_tmp + '/tmp.csv', delimiter, query)
        qtxt = cleandoc(f"""
            USING CHAR SET UTF-8
            DEFINE JOB tdd_{local_seed}
            (
              APPLY TO OPERATOR ($FILE_WRITER)
              SELECT * FROM OPERATOR($EXPORT);
            );
            """)
        with open(path_tmp + '/tdd.txt', 'w+') as f:
            f.write(qtxt)
        with open(path_tmp + '/jobvars.txt', 'w+') as f:
            f.write(txt)
        # Create config file, if missing
        cache_dir = os.path.expanduser('~') + '/cache'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        else:
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        config = cleandoc(f"""
            CheckpointDirectory='{cache_dir}'
            LogDirectory='{cache_dir}'
            """)
        path_config = os.path.expanduser('~') + '/.twbcfg.ini'
        with open(path_config, 'w+') as f:
            f.write(config)
        # Start TPT download
        # p = subprocess.Popen(shlex.split(f"tbuild -f {path_tmp}/qstart2.txt -v {path_tmp}/jobvars.txt -j qstart2"))
        # p.wait()
        result = subprocess.run(
            shlex.split(f"tbuild -f {path_tmp}/tdd.txt -v {path_tmp}/jobvars.txt -j tdd_{str(local_seed)}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Check results
        result = result.stdout.decode('utf-8')
        if 'terminated' in result.lower():
            if not debug:
                shutil.rmtree(path_tmp)
            pprint(result)
            raise Exception('Invalid query.')
        else:
            df = pd.read_csv(path_tmp + '/tmp.csv', header=None,
                             delimiter=delimiter, dtype=dtype, decimal=decimal)
            if obj:
                with teradata.UdaExec(appName='null', version='0', logLevel='ERROR').connect(
                        method="odbc", charset="UTF8", system=schema, username=username, password=password, transactionMode='TERADATA', configureLogging=False) as session:
                    columns = []
                    for row in session.execute(f"help table {obj}"):
                        columns.append(row.values[18].strip())
                df.columns = columns
            if not debug:
                shutil.rmtree(path_tmp)
            if compress:
                return COMPRESS().fit_transform(df, separate_null=compress_separate_null, replace_nulls_inf=compress_replace_nulls_inf, drop_single=compress_drop_single)
            else:
                return df
    else:
        with teradata.UdaExec(appName='null', version='0', logLevel='ERROR').connect(
                method="odbc", charset="UTF8", system=schema, username=username, password=password, transactionMode='TERADATA', configureLogging=False) as session:
            if obj:
                query = f"select * from {obj}"
                for row in session.execute(f"select count(*) from {obj}"):
                    total = int(row.values[0])
            else:
                total = None
            dummies = []
            for row in tqdm(session.execute(query), total=total, disable=not show_progress):
                dummy = {}
                for key, index in row.columns.items():
                    dummy[key] = row.values[index]
                dummies.append(dummy)
        df = pd.DataFrame(dummies)
        if compress:
            return COMPRESS().fit_transform(df, separate_null=compress_separate_null, replace_nulls_inf=compress_replace_nulls_inf, drop_single=compress_drop_single)
        else:
            return df


def td_upload(df, table, schema="tdsb15.cgs.sbrf.ru", username="", password="", batch_size=12000, fast=True,
              max_sessions=6, buffersize=524288, delimiter='^', show_progress=True):
    """Insert pandas.DataFrame into existing table in Teradata of same shape and types

    Parameters
    ----------
    df: pandas.DataFrame
    table: str
        'database.table'
    schema: str, optional
        Teradata schema
    username: str
    password: str
    batch_size: data size in bytes if 'fast' is not specified
    fast: bool, optional
        use Teradata Parallel Transport tool to download data in parallel
    max_sessions: int, optional
        number of parallel jobs if 'fast' is specified
    buffersize: int
        buffer size in bytes if 'fast' is specified
    delimiter: str
        delimiter to use in parsing data
    show_progress: bool, optional
        show progress info

    Returns
    -------
    None
    """
    if '' in [username, password]:
        raise Exception('Empty login or password.')
    if fast:
        if show_progress:
            print("Preparing pd.DataFrame...\r", end='', flush=True)
        # str(random.randint(0, 1000000))
        local_seed = str(round(os.getpid()+time.time()))
        # Create config file, if missing
        cache_dir = os.path.expanduser('~') + '/cache'
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        else:
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        config = cleandoc(f"""
            CheckpointDirectory='{cache_dir}'
            LogDirectory='{cache_dir}'
            """)
        path_config = os.path.expanduser('~') + '/.twbcfg.ini'
        with open(path_config, 'w+') as f:
            f.write(config)
        # Create tmp folder
        path_tmp = os.path.expanduser('~') + '/_' + local_seed
        if os.path.exists(path_tmp):
            shutil.rmtree(path_tmp)
            os.mkdir(path_tmp)
        else:
            os.mkdir(path_tmp)
        # Convert df to string
        converted = df.replace(np.NaN, '').astype(str)
        # Save to temp file
        converted.to_csv(path_tmp + '/tmp.csv', index=False,
                         header=False, sep=delimiter)
        if show_progress:
            print("Uploading to Teradata... \r", end='', flush=True)
        # Get column lengths
        converted_len = converted.apply(
            lambda x: x.str.encode('utf-8').apply(len)).max().to_dict()
        #converted_len = {}
        #for item in Parallel(n_jobs=len(converted.columns))(delayed(column_lengths)(name, series) for name, series in converted.items()):
        #    converted_len.update(item)

        # Create empty table
        td_temp_table = table.split('.')[0] + '.' + next(
            item for item in [os.environ.get('USER').replace('-', '_').replace('.', '_'), 'System'] if item is not None) + local_seed
        td_query(f"create table {td_temp_table} as {table} with no data", schema=schema, username=username,
                 password=password)
        # Create utility file
        txt = cleandoc(f"""
            USING CHARACTER SET UTF8
            DEFINE JOB tdu_{local_seed}
            Description 'Fastload script'
            (
                DEFINE OPERATOR Load_operator
                TYPE LOAD
                SCHEMA *
                ATTRIBUTES
                (
                    VARCHAR TdPid='{schema}',
                    VARCHAR UserName='{username}',
                    VARCHAR UserPassWord='{password}',
                    VARCHAR TargetTable='{td_temp_table}',
                    VARCHAR LogTable='{td_temp_table + '_tpt_log'}',
                    VARCHAR DateForm='AnsiDate',
                    INTEGER MaxSessions={max_sessions}
                );

                DEFINE SCHEMA Define_Employee_Schema
                (
                    {','.join(f'{key} VARCHAR({max(1, value)})' for key, value in converted_len.items())}
                );

                DEFINE OPERATOR Producer_File_Detail
                TYPE DATACONNECTOR PRODUCER
                SCHEMA Define_Employee_Schema
                ATTRIBUTES
                (
                    VARCHAR DirectoryPath='{path_tmp}/'
                    , VARCHAR FileName='tmp.csv'
                    , VARCHAR TextDelimiter='{delimiter}'
                    , VARCHAR Format='Delimited'
                    , VARCHAR OpenMode='Read'
                    , VARCHAR INDICATORMODE='N'
                    , INTEGER BUFFERSIZE = {buffersize}
                );

                APPLY
                (
                   'INSERT INTO {td_temp_table}({','.join(f'{key}' for key, value in converted_len.items())}) VALUES (:{',:'.join(f'{key}' for key, value in converted_len.items())});'
                )
                TO OPERATOR(Load_operator)

                SELECT * FROM OPERATOR (Producer_File_Detail);
            );
            """)
        with open(path_tmp + '/load_code.tpt', 'w+') as f:
            f.write(txt)
        # Start TPT load
        result = subprocess.run(
            shlex.split(f"tbuild -f {path_tmp}/load_code.tpt -L {path_tmp}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        # Check results
        result = result.stdout.decode('utf-8')
        if 'terminated' in result.lower():
            shutil.rmtree(path_tmp)
            pprint(result)
            raise Exception('Invalid query.')
        else:
            # Insert temporary table into main
            if show_progress:
                print("Merging in Teradata...   \r", end='', flush=True)
            td_query(f"insert into {table} sel * from {td_temp_table}", schema=schema, username=username,
                     password=password)
            # Drop temporary table
            if show_progress:
                print("Cleaning...              \r", end='', flush=True)
            td_query(f"drop table {td_temp_table}", schema=schema,
                     username=username, password=password)
            # Cleanup
            shutil.rmtree(path_tmp)
            if show_progress:
                print("Uploaded!                    ")
    else:
        n_iters = len(df) // batch_size + (len(df) % batch_size > 0)
        #df_dict = df.to_dict('records')
        with teradata.UdaExec(appName='null', version='0', logLevel='ERROR').connect(
                method="odbc", charset="UTF8", system=schema, username=username, password=password, transactionMode='TERADATA') as session:
            for i in tqdm(range(n_iters), total=n_iters):
                session.executemany(
                    # , [list(row.values()) for row in df_dict[i * batch_size:i * batch_size + batch_size]]
                    f"INSERT INTO {table} VALUES ({','.join(list('?'*df.shape[1]))})", list(df[i * batch_size:i * batch_size + batch_size].itertuples(index=False, name=None)), batch=True)


def td_proc(name, params=(), schema="tdsb15.cgs.sbrf.ru", username="", password=""):
    """Execute Teradata procedure with options, if needed. Doesn't support output.

    Parameters
    ----------
    name: str
        'database.procedure_name'
    params: list
        list of procedure parameters, beware of types
    schema: str, optional
        Teradata schema
    username: str
    password: str

    Returns
    -------
    None
    """
    with teradata.UdaExec(appName='null', version='0', logLevel='ERROR').connect(
            method="odbc", charset="UTF8", system=schema, username=username, password=password, transactionMode='TERADATA', configureLogging=False) as session:
        session.callproc(name, params=params)


def td_query(query, schema="tdsb15.cgs.sbrf.ru", username="", password=""):
    """Executes custom query and prints result in output

    Parameters
    ----------
    query:
    schema: str, optional
        Teradata schema
    username: str
    password: str

    Returns
    -------
    None, prints query result in output
    """
    with teradata.UdaExec(appName='null', version='0', logLevel='ERROR').connect(
            method="odbc", charset="UTF8", system=schema, username=username, password=password, transactionMode='TERADATA', configureLogging=False) as session:
        for row in session.execute(query):
            pprint(row.values[0])


def kerb_ticket(username=None, password=None, server='momos11.ca.sbrf.ru'):
    """Download updated Kerberos ticket from momos

    Parameters
    ----------
    username: str
        momos username
    password: str
        momos password
    server: str, optional
        One of Hadoop nodes

    Returns
    -------
    str, local path to Kerberos ticket
    """

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(server, username=username, password=password)
    ssh_stin, ssh_stdout, ssh_stderr = ssh.exec_command('klist')
    remote_ticket = ssh_stdout.read().decode(
        'utf-8').split('\n')[0].split('FILE:')[1]

    local_kerb = os.path.join(os.path.expanduser('~'), '.ssh')
    if not os.path.exists(local_kerb):
        os.mkdir(local_kerb)
    local_kerb_ticket = os.path.join(local_kerb, remote_ticket.split('/')[2])
    if not os.path.exists(local_kerb_ticket) or os.path.getsize(local_kerb_ticket)==0:
        Path(local_kerb_ticket).touch(mode=0o600)
        with ssh.open_sftp() as ftp:
            ftp.get(remote_ticket, local_kerb_ticket)
    ssh.close()

    return local_kerb_ticket


def train_val_test_split(data, fractions=[60, 20, 20], random_state=1, shuffle=False):
    """Splits dataset into three ones

    Parameters
    ----------
    data: indexable
        object to split
    fractions: list, optional
        list with percentages
    random_state: int, optional
    shuffle: bool, optional
        whether to shuffle the data before splitting

    Returns
    -------
    tuple(train, val, test)
    """
    train, tmp = train_test_split(data, train_size=fractions[0] / 100, random_state=random_state, shuffle=shuffle)
    val, test = train_test_split(tmp, train_size=fractions[1] / 100, random_state=random_state, shuffle=shuffle)

    return train, val, test


def tokenize(series, categories=None):
    """Create forward/backward dictionaries to tokenise pd.Series

    Parameters
    ----------
    series: pd.Series
        object to tokenize
    categories: list, optional
        list of values to count as separate levels, all left ones will be replaced by '-1' token

    Returns
    -------
    tuple(dict, dict)
    """
    unique = pd.Series(series.unique())
    if not categories:
        categories = unique[~pd.isnull(unique)]
    # TODO: don't process full series, just unique values
    tokenized = unique.astype(CategoricalDtype(categories=categories)).cat.codes
    token_item = dict(zip(tokenized, unique))
    item_token = {v: k for k, v in token_item.items()}

    return token_item, item_token


def heatmap(x, y, color=None, palette=None, color_range=None, size=None, size_range=None, size_scale=500, x_order=None
            , y_order=None, marker='s', title=''):
    if color is None:
        color = [1] * len(x)
    if palette:
        n_colors = len(palette)
    else:
        n_colors = 256  # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors)

    if color_range:
        color_min, color_max = color_range
    else:
        color_min, color_max = min(color), max(
            color)  # Range of values that will be mapped to the palette, i.e. min and max possible correlation
    if size is None:
        size = [1] * len(x)

    if size_range:
        size_min, size_max = size_range
    else:
        size_min, size_max = min(size), max(size)

    if x_order:
        x_names = [t for t in x_order]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]: p[0] for p in enumerate(x_names)}

    if y_order:
        y_names = [t for t in y_order]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]: p[0] for p in enumerate(y_names)}

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (
                        color_max - color_min)  # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1)  # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1))  # target index in the color palette
            return palette[ind]

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (
                        size_max - size_min) + 0.01  # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1)  # bound the position betwen 0 and 1
            return val_position * size_scale

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)  # Setup a 1x10 grid
    ax = plt.subplot(plot_grid[:, :-1])  # Use the left 14/15ths of the grid for the main plot
    plt.title(title)

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size],
        c=[value_to_color(v) for v in color],
    )
    ax.set_xticks([v for k, v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k, v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:, -1])  # Use the rightmost column of the plot

        col_x = [0] * len(palette)  # Fixed x coordinate for the bars
        bar_y = np.linspace(color_min, color_max, n_colors)  # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5] * len(palette),  # Make bars 5 units wide
            left=col_x,  # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2)  # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False)  # Hide grid
        ax.set_facecolor('white')  # Make background white
        ax.set_xticks([])  # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))  # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right()  # Show vertical ticks on the right


def confusion_plot(y_true, y_pred, figsize=(6,5.5), log=False, **kwargs):
    # Confusion matrix in logscale
    cmatrix = confusion_matrix(y_true, y_pred)
    index = sorted(set(y_true) | set(y_pred))
    if log:
        cmatrix = np.where(cmatrix>0, np.log(cmatrix), 0)
    else:
        cmatrix = np.where(cmatrix>0, cmatrix, 0)
    cmatrix_melted = pd.melt(pd.DataFrame(cmatrix, columns=index, index=index).reset_index(), id_vars='index')
    cmatrix_melted.columns = ['x', 'y', 'value']