import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from ciber.analytics import stats as pstats
import misc as put

CMAP = sns.diverging_palette(20, 220, as_cmap=True)

def heatmap(df, ax=None, figsize=(10, 5), title=None, annot=True, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title, y=1)
    else:
        ax.set_title(title)
    sns.heatmap(df, annot=annot, cmap=CMAP, ax=ax, cbar=False, **kwargs)
    ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()


def get_ax(ax=None, **kwargs):
    if ax is None:
        plt.figure(**kwargs)
        ax = plt.gca()
    return ax


def annotate(ax=None, **kwargs):
    ax = get_ax(ax)
    for container in ax.containers:
        ax.bar_label(container, **kwargs)


def scale_figsize(n_features, one_figsize=(5, 3), n_col=3):
    n_row = int(np.ceil(n_features / n_col))
    figsize = (one_figsize[0] * n_col, one_figsize[1] * n_row)
    layout = (n_row, n_col)
    return layout, figsize


def legend_outside(fig=None, handles=None, labels=None, ax=None,
                   style='long', ncol='auto', max_dim=12, bbox_to_anchor=None, **kwargs):
    """
    :param ax:
    :param style:
    :param ncol:
    :param max_dim: if style='long', then max_row=12; if style='wide', then max_col=12
    :return:
    """
    assert style in ['long', 'wide']


    if ax is not None:
        h, w = ax.bbox.height, ax.bbox.width
        n_label = len(ax.get_legend_handles_labels()[0])
    else:
        assert handles is not None and labels is not None
        h, w = fig.bbox.height, fig.bbox.width
        n_label = len(labels)

    if n_label == 0:
        return

    if style == 'long':
        loc = 'center left'
        if ncol == 'auto':
            ncol = max(1, int(np.ceil(n_label / max_dim)))
        if bbox_to_anchor is None:
            bbox_to_anchor = (1, 0.5)
    else:
        loc='upper center'
        if ncol == 'auto':
            ncol = min(n_label, max_dim)
        if bbox_to_anchor is None:
            bbox_to_anchor = (0.5, -100 / h)

    kws = {'loc': loc, 'bbox_to_anchor': bbox_to_anchor,
           'fancybox': True, 'ncol': ncol, **kwargs}
    if ax is not None:
        return ax.legend(**kws)
    return fig.legend(handles, labels, **kws)


def data_quality_check(df, disc_cols=[], sparse_cols=[], check_zero=False, miss_thres=0., one_figsize=(15, 5)):
    ncols = 2
    if len(disc_cols) > 0:
        ncols += 1
    if len(sparse_cols) > 0:
        ncols += 1
    layout, figsize = scale_figsize(ncols, one_figsize=one_figsize, n_col=1)
    fig, ax = plt.subplots(*layout, figsize=figsize)

    df_dense = df.drop(columns=sparse_cols)
    is_missing = df_dense.isna()
    if check_zero:
        is_missing = is_missing | (df_dense._get_numeric_data().abs() < 1e-6)
    missing_pct = is_missing.mean().sort_values()
    missing_pct = missing_pct[missing_pct >= miss_thres]
    missing_pct.plot.barh(title='Percentage of Missing Values', ax=ax[0], xlim=(0, 1))
    annotate(ax[0], fmt="{:.0%}")

    nuni = df_dense.nunique().sort_values(ascending=False)
    nuni.drop(disc_cols).plot.barh(title='Number of Unique Values (Cont. Cols)', ax=ax[1])
    annotate(ax[1])

    if len(disc_cols) > 0:
        nuni[nuni.index.isin(disc_cols)].plot.barh(title='Number of Unique Values (Disc. Cols)', ax=ax[2])
        annotate(ax[2])

    if len(sparse_cols) > 0:
        sparse_counts = df[sparse_cols].replace(0.0, np.nan).notna().mean().sort_values(ascending=True)
        sparse_counts.plot.barh(title='%Count of Sparse Cols', ax=ax[-1])
        annotate(ax[-1], fmt="{:.0%}")

    fig.tight_layout()
    plt.show()



def plot_corr(corr, dist=False, power=2,
              cluster=True, linkage_method='complete', optimal_ordering=True,
              corr_thres=None, annot=True, title=None, **kwargs):
    mask = put.is_diag(corr)
    center = 100 if dist else 0
    kwargs = {**{"fmt": '.0f', 'center': center, 'vmin': -100, 'vmax': 100}, **kwargs}
    dist_mat, dist_mat_filled = pstats.corr2dist(corr, power)

    if corr_thres is not None:
        mask = mask | (corr.abs() <= corr_thres)

    to_show = dist_mat if dist else corr
    if not cluster:
        return heatmap(to_show.mask(mask) * 100, figsize=(10, 10), title=title, **kwargs)

    Z = linkage(squareform(dist_mat_filled), linkage_method, optimal_ordering=optimal_ordering)
    g = sns.clustermap(to_show * 100, row_linkage=Z, col_linkage=Z,
                       mask=mask, annot=annot, cmap=CMAP, **kwargs)
    if title is not None:
        g.ax_col_dendrogram.set_title(title)
    return g


def plot_contingency(ctg_tab, sort=True, one_figsize=(8, 8), corr=False, min_marg_prob=0.01, min_cond_prob=0.1,
                     xname=None, yname=None, title=None, n_col=2, **kwargs):
    if xname is None:
        xname = ctg_tab.axes[0].name
    if yname is None:
        yname = ctg_tab.axes[1].name

    tab = sm.stats.Table(ctg_tab)
    if corr:
        # It shows the Dummy Correlation:
        # Corr(X=i, Y=j) = (p_ij - p_i * p_j) / sqrt(p_i * (1 - p_i) * p_j * (1 - p_j))
        to_show = tab.standardized_resids / np.sqrt(tab.table.sum()) * 100
    else:
        to_show = ctg_tab.copy()

    row_pct, col_pct = tab.marginal_probabilities
    # row_pct, col_pct = row_pct[row_pct >= min_prob], col_pct[col_pct >= min_prob]

    if len(row_pct) == 0:
        print(f'Categories {xname} too sparse, skipping plot!')
        return

    if len(col_pct) == 0:
        print(f'Categories {yname} too sparse, skipping plot!')
        return

    if sort:
        row_pct = row_pct.sort_values(ascending=False)
        col_pct = col_pct.sort_values(ascending=False)

    to_show = to_show.loc[row_pct.index, col_pct.index]
    to_show.index += row_pct.apply(lambda x: f' ({x * 100:.0f}%)').values
    to_show.columns += col_pct.apply(lambda x: f' ({x * 100:.0f}%)').values

    if corr:
        return heatmap(to_show, figsize=one_figsize,
                       center=0, fmt='.0f', title=title, vmin=-100, vmax=100, **kwargs)

    row_norm = to_show.T.div(to_show.sum(axis=1)) * 100
    row_norm = row_norm.iloc[:, np.where(row_pct >= min_marg_prob)[0]]
    row_norm = row_norm[(row_norm >= min_cond_prob * 100).any(axis=1)]

    col_norm = to_show.div(to_show.sum()) * 100
    col_norm = col_norm.iloc[:, np.where(col_pct >= min_marg_prob)[0]]
    col_norm = col_norm[(col_norm >= min_cond_prob * 100).any(axis=1)]

    layout, figsize = scale_figsize(2, one_figsize=one_figsize, n_col=n_col)
    fig, ax = plt.subplots(*layout, figsize=figsize)

    heatmap(row_norm, center=0, vmax=100, fmt='.0f', ax=ax[0], title=f'Normalized by {xname}', **kwargs)
    heatmap(col_norm, center=0, vmax=100, fmt='.0f', ax=ax[1], title=f'Normalized by {yname}', **kwargs)
    fig.suptitle(title, y=1)
    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_categ_stats(categ_series, categ_name=None, n_bucket=30,
                     max_grp_size=5, max_str_size=30,
                     title=None, figsize=(15, 10), **kwargs):
    if categ_name is None:
        categ_name = categ_series.name or 'categ'

    categ_series = categ_series.astype(str)
    categ_series[categ_series.str.len() > max_str_size] = categ_series.str[:max_str_size] + '...'
    # Get a pct count for each category
    cate_count = categ_series.value_counts().sort_values(ascending=False)
    cate_count_ratio = cate_count.rename('pct') / cate_count.sum()

    # Split categ according to n_bucket
    big_categ = cate_count_ratio[cate_count_ratio >= 1/n_bucket]
    small_categ = cate_count_ratio[cate_count_ratio < 1/n_bucket]
    buckets = np.ceil(small_categ.cumsum() * n_bucket).astype(int)

    # For small categories, group them, get the grouped name and pct
    small_categ_grp_name = small_categ.groupby(buckets).apply(lambda s: ', '.join(s.index))
    small_categ_grp_pct = small_categ.groupby(buckets).sum()
    small_categ_grp_size = small_categ.groupby(buckets).count()
    small_categ_stats = pd.concat({'group': small_categ_grp_name,
                                   'pct': small_categ_grp_pct,
                                   'group_size': small_categ_grp_size}, axis=1)

    # If the group_size is too large, then we only show the group_size on x-axis
    small_categ_stats = small_categ_stats.set_index('pct')
    small_categ_stats[categ_name] = np.where(small_categ_stats.group_size > max_grp_size,
                                             small_categ_stats.group_size.apply(lambda s: f'group_size={s}'),
                                             small_categ_stats.group)
    small_categ_stats = small_categ_stats.reset_index().set_index(categ_name)['pct']

    if title is None:
        title = f'% Count of Categories for {categ_name}, # Unique={len(cate_count)}'
    categ_stats = pd.concat([big_categ, small_categ_stats])
    return categ_stats[::-1].plot.barh(title=title, figsize=figsize, **kwargs)


def plot_surface(df, figsize=(15, 10), title='',
                 xlabel=None, ylabel=None, contour=True,
                 aspect=(1, 1, 1), contour_margins=(0.25, 0.5, 1),
                 elev=30, azim=-135):
    """
    :param df: should be grid-like, index will be 'x', columns will be 'y'
    :param figsize:
    :param title:
    :param xlabel:
    :param ylabel:
    :param aspect:
    :param contour_margins:
    :return:
    """

    xlabel = xlabel or df.index.name or "x"
    ylabel = ylabel or df.columns.name or "y"

    x, y = df.index, df.columns
    X, Y = np.meshgrid(x, y)
    Z = df.T.values

    ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=CMAP)
    ax.set_box_aspect(aspect=aspect)

    kwargs = {}
    if contour:
        for dim, var, offset_dir, margin in zip(['x', 'y', 'z'], [x, y, Z], ['-', '+', '-'], contour_margins):
            var_min, var_max = np.nanmin(var), np.nanmax(var)
            var_range = var_max - var_min

            if offset_dir == '-':
                offset = var_min - margin * var_range
                kwargs[f'{dim}lim'] = (offset, var_max)
            else:
                offset = var_max + margin * var_range
                kwargs[f'{dim}lim']  = (var_min, offset)

            kwargs[f'{dim}ticks'] = ax.__getattribute__(f'get_{dim}ticks')()
            ax.contourf(X, Y, Z, zdir=dim, offset=offset, cmap=CMAP)

    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, **kwargs)
    ax.view_init(elev=elev, azim=azim)
    return ax

def points2surf(X, y):
    assert len(X) == len(y)
    d = X.shape[1]
    df = pd.Series(y.flatten(), index=pd.MultiIndex.from_arrays(X.T))
    df = df.rename_axis([f'x_{i+1}' for i in range(d)]).rename('y')
    df = df.unstack(level=list(range(1, df.index.nlevels))) # Only keep x_1 as index
    return df

def surf2points(df):
    return df.stack(level=list(range(df.columns.nlevels))).rename('y').reset_index()

def scatter_3d(df, ax=None, figsize=(15, 10), title='',
                 xlabel=None, ylabel=None,
                 aspect=(1, 1, 1), contour_margins=(0.25, 0.5, 1),
                 elev=30, azim=315, **kwargs):
    """
    :param df: should be grid-like, index will be 'x', columns will be 'y'
    :param figsize:
    :param title:
    :param xlabel:
    :param ylabel:
    :param aspect:
    :param contour_margins:
    :return:
    """
    p = surf2points(df)
    xlabel = xlabel or df.index.name or "x"
    ylabel = ylabel or df.columns.name or "y"

    if ax is None:
        ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
        ax.set_box_aspect(aspect=aspect)

        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)#, **kwargs)
        ax.view_init(elev=elev, azim=azim)

    if kwargs.get('color') is None and kwargs.get('c') is None:
        kwargs['c'] = p.values[:, -1]
        kwargs['cmap'] = CMAP

    ax.scatter3D(*p.values.T, **kwargs)
    return ax