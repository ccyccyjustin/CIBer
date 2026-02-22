import pandas as pd
import numpy as np
from collections import defaultdict
import logging
from joblib import Parallel, delayed
from itertools import combinations
from tqdm import tqdm
logger = logging.getLogger(__name__)


def parallel(func, iterable, n_jobs=-1, disable_par=False, **kwargs):
    if disable_par:
        results = {i: func(i, **kwargs) for i in tqdm(iterable)}
        return results

    results = Parallel(n_jobs=n_jobs)(
        delayed(func)(i, **kwargs) for i in tqdm(iterable)
    )
    results = {k: v for k, v in zip(iterable, results)}
    return results


def is_diag(X):
    matrix = np.zeros(X.shape, dtype=bool)
    np.fill_diagonal(matrix, True)
    return matrix


def is_iter_not_str(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, str)


def swap(d):
    """
    Swap (key, value) pair, if value contains duplicates, then will be stored as a list
    :param d:
    :return:
    """
    d2 = {}
    for k, v in d.items():
        if v not in d2:
            d2[v] = [k]
        else:
            d2[v].append(k)
    if all([len(v) == 1 for v in d2.values()]):
        d2 = {k: v[0] for k, v in d2.items()}
    return d2