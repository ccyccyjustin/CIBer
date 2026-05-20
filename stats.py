# -------------------- MUTUAL INFORMATION AND DISTANCE CORRELATION --------------------
import numpy as np
import pandas as pd
from sklearn.feature_selection._mutual_info import *
from sklearn.feature_selection._mutual_info import _compute_mi, _iterate_columns
from scipy.spatial.distance import squareform
from itertools import combinations
import util as put
import plot as pplot

from _dcorr import distance_correlation
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

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

def plot_silhouette(X, cluster_labels, title='Silhouette Analysis',
                    metric='euclidean', cluster_keys=None, copy=True, ax=None, figsize=None,
                    cmap='nipy_spectral', title_fontsize="large",
                    text_fontsize="medium"):
    """Plots silhouette analysis of clusters provided."""
    cluster_labels = np.asarray(cluster_labels)

    le = LabelEncoder()
    cluster_labels_encoded = le.fit_transform(cluster_labels)

    n_clusters = len(np.unique(cluster_labels))

    silhouette_avg = silhouette_score(X, cluster_labels, metric=metric)

    sample_silhouette_values = silhouette_samples(X, cluster_labels,
                                                  metric=metric)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if cluster_keys is None:
        cluster_keys = {i: i for i in np.unique(cluster_labels)}

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlim([-0.1, 1])

    ax.set_ylim([0, len(X) + (n_clusters + 1) * 10 + 10])

    ax.set_xlabel('Silhouette coefficient values', fontsize=text_fontsize)
    ax.set_ylabel('Cluster label', fontsize=text_fontsize)

    y_lower = 10

    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[
            cluster_labels_encoded == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.get_cmap(cmap)(float(i) / n_clusters)

        ax.fill_betweenx(np.arange(y_lower, y_upper),
                         0, ith_cluster_silhouette_values,
                         facecolor=color, edgecolor=color, alpha=0.7,
                         label=str(le.classes_[i]) + ': ' + cluster_keys[le.classes_[i]])# + '\n')

        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(le.classes_[i]),
                fontsize=text_fontsize)

        y_lower = y_upper + 10

    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
               label='Silhouette score: {0:0.3f}'.format(silhouette_avg))

    ax.set_yticks([])  # Clear the y-axis labels / ticks
    ax.set_xticks(np.arange(-0.1, 1.0, 0.2))

    ax.tick_params(labelsize=text_fontsize)
    ax.legend(reverse=True, loc='best', fontsize=text_fontsize)
    return ax


def optimal_feature_agglomeration(dist_matrix: pd.DataFrame, linkage='complete', min_k=2, max_k=None,
                                  early_stopping=False, **kwargs):
    """
    Finds the optimal FeatureAgglomeration model based on Silhouette Score.

    Parameters:
    dist_matrix: ndarray of shape (n_features, n_features)
    min_k: Minimum number of clusters to check
    max_k: Maximum number of clusters to check (defaults to n_features - 1)
    """
    n_features = dist_matrix.shape[0]
    if max_k is None:
        max_k = n_features - 1

    best_score = -1
    best_model = None

    scores = {}
    # Iterate through possible cluster counts
    for k in range(min_k, max_k + 1):
        model = FeatureAgglomeration(n_clusters=k, metric='precomputed', linkage=linkage, **kwargs)

        # Fit using the distance matrix
        model.fit(dist_matrix)
        labels = model.labels_

        # Calculate silhouette score
        score = silhouette_score(dist_matrix, labels, metric='precomputed', linkage=linkage, **kwargs)
        scores[k] = score

        if score > best_score:
            best_score = score
            best_model = model
        elif early_stopping:
            break
    return best_model

def feature_agglomeration(distance_matrix, n_clusters=None, min_asso=None,
                          linkage='complete', power=1, early_stopping=False, **kwargs):
    if n_clusters is None and min_asso is None:
        AGNES = optimal_feature_agglomeration(distance_matrix, linkage, early_stopping=early_stopping)
    else:
        if n_clusters is not None:
            distance_threshold = None
        else:
            distance_threshold = np.sqrt(1 - min_asso ** power)
        # TODO: show similarity matrix after clustering
        AGNES = FeatureAgglomeration(metric='precomputed', linkage=linkage,
                                     distance_threshold=distance_threshold, n_clusters=n_clusters)
        AGNES.fit(distance_matrix)

    best_labels = AGNES.labels_
    best_k = AGNES.n_clusters_

    clusters = put.swap(dict(zip(AGNES.feature_names_in_, best_labels)))
    cluster_keys = {k: str(v).replace(', ', ',\n') for k, v in clusters.items()}
    ax = plot_silhouette(distance_matrix, best_labels, metric='precomputed', cluster_keys=cluster_keys,
                    title=f'Silhouette Analysis for Selected K={best_k}, linkage={linkage}', **kwargs)
    pplot.legend_outside(ax=ax, reverse=True, labelspacing=1.0)
    return clusters