# -------------------- MUTUAL INFORMATION AND DISTANCE CORRELATION --------------------
import numpy as np
import pandas as pd
from sklearn.feature_selection._mutual_info import *
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from scipy.spatial.distance import squareform
from itertools import combinations
import util as put

def estimate_mi(
        X,
        *,
        discrete_features="auto",
        n_neighbors=3,
        copy=True,
        random_state=None,
        n_jobs=None,
):
    X = check_array(X, accept_sparse="csc", input_name='X')
    n_samples, n_features = X.shape

    if isinstance(discrete_features, (str, bool)):
        if isinstance(discrete_features, str):
            if discrete_features == "auto":
                discrete_features = issparse(X)
            else:
                raise ValueError("Invalid string value for discrete_features.")
        discrete_mask = np.empty(n_features, dtype=bool)
        discrete_mask.fill(discrete_features)
    else:
        discrete_features = check_array(discrete_features, ensure_2d=False)
        if discrete_features.dtype != "bool":
            discrete_mask = np.zeros(n_features, dtype=bool)
            discrete_mask[discrete_features] = True
        else:
            discrete_mask = discrete_features

    continuous_mask = ~discrete_mask
    if np.any(continuous_mask) and issparse(X):
        raise ValueError("Sparse matrix `X` can't have continuous features.")

    rng = check_random_state(random_state)
    if np.any(continuous_mask):
        X = X.astype(np.float64, copy=copy)
        X[:, continuous_mask] = scale(
            X[:, continuous_mask], with_mean=False, copy=False
        )

        # Add small noise to continuous features as advised in Kraskov et. al.
        means = np.maximum(1, np.mean(np.abs(X[:, continuous_mask]), axis=0))
        X[:, continuous_mask] += (
                1e-10
                * means
                * rng.standard_normal(size=(n_samples, np.sum(continuous_mask)))
        )
    def _safe_compute_mi(x, y, discrete_x, discrete_y, n_neighbors):
        try:
            return _compute_mi(x, y, discrete_x, discrete_y, n_neighbors)
        except:
            return np.nan

    mi = Parallel(n_jobs=n_jobs)(
        delayed(_safe_compute_mi)(x, y, discrete_x, discrete_y, n_neighbors)
        for (x, discrete_x), (y, discrete_y) in combinations(zip(_iterate_columns(X), discrete_mask), 2)
    )

    mi = np.array(mi)
    return mi


def mi_corr(X, discrete_features='auto', **kwargs):
    mi = estimate_mi(X, discrete_features=discrete_features, **kwargs)
    mi = squareform(mi)
    mi_corr = np.sqrt(1 - np.exp(-2 * mi))
    np.fill_diagonal(mi_corr, 1)
    if isinstance(X, pd.DataFrame):
        mi_corr = pd.DataFrame(mi_corr, index=X.columns, columns=X.columns)
    return mi_corr


def corr2dist(corr, power=2):
    dist_mat = np.sqrt(1 - np.abs(corr) ** power)
    dist_mat_filled = dist_mat.copy()
    mask = put.is_diag(corr)
    dist_mat_filled[mask] = 0
    dist_mat_filled = dist_mat_filled.fillna(1)
    dist_mat_filled = (dist_mat_filled + dist_mat_filled.T) / 2
    return dist_mat, dist_mat_filled


def pcorr(corr_mat):
    prec_mat = corr_mat * 0 + np.linalg.inv(corr_mat)
    p_sqrt = np.sqrt(np.diag(prec_mat)).reshape(-1, 1)
    pcorr_mat = - prec_mat / (p_sqrt @ p_sqrt.T)
    return pcorr_mat


def corr_eigen(corr_mat, exp_var=1.0):
    eigval, eigvec = np.linalg.eigh(corr_mat)

    eigval = pd.Series(eigval[::-1], index=range(1, corr_mat.shape[1] + 1))
    eigval = eigval / eigval.sum()

    eigvec = pd.DataFrame(eigvec[:, ::-1], index=corr_mat.columns, columns=range(1, corr_mat.shape[1] + 1))
    eigvec = eigvec.mul(np.sign(eigvec.sum(axis=0)), axis=1)

    large_pcs = eigval[eigval.cumsum() <= exp_var].index
    return eigval[large_pcs], eigvec[large_pcs]

