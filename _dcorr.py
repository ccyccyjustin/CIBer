from numba import njit, prange, float64, boolean
from itertools import combinations_with_replacement
from time import time
from scipy.stats import energy_distance
from dcor import distance_covariance_sqr
import numpy as np
import pandas as pd

import util


@njit(float64(float64[:], float64[:], boolean, boolean), parallel=True)
def _dcov(x, y, x_disc=False, y_disc=False):
    sumprod_xy, sum_x, sum_y, n_valid = 0., 0., 0., 0.
    n = len(x)

    def dist(a, b, disc, tol=1e-8):
        d = np.abs(a - b)
        if not disc:
            return d
        if d < tol:
            return 0.
        return 1.

    def dist_mean(arr, arr_other, idx, disc):
        # Compute AVG distance of arr_i with others
        dsum_i, ni = 0., 0.
        for k in prange(n):
            if np.isnan(arr[k]) or np.isnan(arr_other[k]):
                continue
            dsum_i += dist(arr[idx], arr[k], disc)
            ni += 1.
        dmean_i = dsum_i / ni
        return dmean_i

    for i in prange(n):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue

        dmean_xi = dist_mean(x, y, i, x_disc)
        dmean_yi = dist_mean(y, x, i, y_disc)

        for j in prange(n):
            if np.isnan(x[j]) or np.isnan(y[j]):
                continue

            dx = dist(x[i], x[j], x_disc)
            dy = dist(y[i], y[j], y_disc)

            sumprod_xy += (dx * dy - dmean_xi * dy - dx * dmean_yi)
            sum_x += dx
            sum_y += dy
            n_valid += 1.

    cov = sumprod_xy / n_valid
    cov += (sum_x / n_valid) * (sum_y / n_valid)
    return cov


def dcov(x, y, x_disc=False, y_disc=False):
    assert x.shape[0] == y.shape[0], "x, y different shape!"
    assert x.ndim == y.ndim == 1, "x, y must be 1d!"
    x, y = np.array(x).astype(np.float64), np.array(y).astype(np.float64)
    return _dcov(x, y, x_disc, y_disc)


def pair_dropna(x, y):
    good_pair = np.isfinite(x) & np.isfinite(y)
    return x[good_pair], y[good_pair]


def fast_cc_dcov(x_cont, y_cont):
    x2, y2 = pair_dropna(x_cont, y_cont)
    return distance_covariance_sqr(x2, y2)


def fast_dd_dcov(x_disc, y_disc):
    ctg_tab = pd.crosstab(x_disc, y_disc, normalize='all')
    col_prob = ctg_tab.sum(axis=0).values
    row_prob = ctg_tab.sum(axis=1).values
    cov = np.sum((ctg_tab.values - row_prob[:, None] * col_prob[None, :]) ** 2)
    return cov


def fast_cd_dcov(x_cont, y_disc, min_prob=0.01):
    x2, y2 = pair_dropna(x_cont, y_disc)
    unique_values, counts = np.unique(y2, return_counts=True)
    y_prob = counts / np.sum(counts)

    # edists=0 for small categories
    large_cat = y_prob >= min_prob
    if sum(large_cat) == 0:
        return 0. #np.nan

    unique_values, y_prob = unique_values[large_cat], y_prob[large_cat]
    edists = np.array([energy_distance(x2[y2 == c], x2) for c in unique_values])
    dcov = np.sum((edists * y_prob) ** 2)
    return dcov


def fast_dc_dcov(x_disc, y_cont):
    return fast_cd_dcov(y_cont, x_disc)


def distance_correlation(X: pd.DataFrame, disc_features=[], rank=True, debug=False, **kwargs):
    if rank:
        X = X.copy()
        cont_features = X.columns[~X.columns.isin(disc_features)]
        X[cont_features] = X[cont_features].rank(pct=True)

    FAST_FUNCS = {(False, False): fast_cc_dcov,
                  (False, True): fast_cd_dcov,
                  (True, False): fast_dc_dcov,
                  (True, True): fast_dd_dcov}

    def _run_one(columns):
        col1, col2 = columns
        x_disc = col1 in disc_features
        y_disc = col2 in disc_features

        t0 = time()
        dc = FAST_FUNCS[(x_disc, y_disc)](X[col1], X[col2])
        t1 = time()

        if debug:
            t2 = time()
            dc_slow = dcov(X[col1], X[col2], x_disc, y_disc)
            t3 = time()
            print(col1, col2, "fast time: ", t1-t0, "slow_time: ", t3 - t2, "error", dc_slow-dc)

        if dc is None or dc < -1e-4:
            raise ValueError("Negative dCov: ", dc, X[[col1, col2]])
        return max(dc, 0)

    cov_mat = util.parallel(_run_one, list(combinations_with_replacement(X, 2)), **kwargs)
    cov_mat = pd.Series(cov_mat).unstack()
    cov_mat = cov_mat.loc[X.columns, X.columns]
    cov_mat = cov_mat.fillna(cov_mat.T)

    x_var = np.diag(cov_mat)
    bad_cols = np.abs(x_var[:, None] + x_var[None, :]) < 1e-8
    cov_mat[bad_cols] = np.nan

    corr_mat = cov_mat / np.sqrt(x_var[:, None] * x_var[None, :])
    corr_mat = np.sqrt(corr_mat)
    return corr_mat