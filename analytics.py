import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
from sklearn.cluster import FeatureAgglomeration, AgglomerativeClustering
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from imblearn.over_sampling import SMOTENC

import util as put
import plot as pplot
import stats as pstats


SCALERS = {'Standard': StandardScaler,
           'MinMax': MinMaxScaler,
           'MaxAbs': MaxAbsScaler,
           'Robust': RobustScaler}


def split_data(df, target, cont_cols, regr=False, max_batch_size=None, test_size=0.2, q_clip=0.,
               scaler=False, oversample=False, balanced=False, random_state=None):
    # Split Train and Test Data
    if max_batch_size is not None:
        assert max_batch_size <= df.shape[0]
        df = df.sample(max_batch_size, random_state=random_state)
    cont_cols = [c for c in cont_cols if c != target]
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Clip and Normalize Continuous Columns
    X_train_cont, X_test_cont = X_train[cont_cols], X_test[cont_cols]
    lb, ub = X_train_cont.quantile(q_clip), X_train_cont.quantile(1 - q_clip)
    X_train_cont, X_test_cont = X_train_cont.clip(lb, ub, axis=1), X_test_cont.clip(lb, ub, axis=1)

    if regr:
        y_train = y_train.clip(*y_train.quantile([q_clip, 1 - q_clip]))

    # Scale Features
    if scaler:
        assert scaler in SCALERS
        scaler = SCALERS[scaler]()
        X_train_cont = scaler.fit_transform(X_train_cont)
        X_test_cont = scaler.transform(X_test_cont)

    X_train[cont_cols] = X_train_cont
    X_test[cont_cols] = X_test_cont

    # Over-sample Data for Classification Problems
    if oversample and not regr:
        categorical_features = [col for col in df.columns if (col not in cont_cols) and (col != target)]
        oversample = SMOTENC(categorical_features=categorical_features, random_state=random_state)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

    if balanced and not regr:
        sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    else:
        sample_weight = None

    return X_train, X_test, y_train, y_test, sample_weight


class DataAnalyzer(object):
    def __init__(self, df, target, cont_thres='log2', sparse_cols=[], verbose=True, random_state=None):
        self.df_orig = df
        self.target = target
        self.df = self.split_feature_target(df, target)

        self.sparse_cols = sparse_cols
        self.df_dense = self.df.drop(columns=self.sparse_cols)
        self.cont_cols = None
        self.disc_cols = None

        self.regr = None
        self.get_cont_features(thres=cont_thres)

        self.idx2categ = {}
        self.categ_map = {}
        self.encode()
        self.df_keep_disc_na = self.keep_disc_na()

        if verbose:
            self.print_col_info()
        self.similarity_matrix = {}
        self.random_state = random_state

    # -------------------- BASIC PROCESSING --------------------
    def split_feature_target(self, df, target):
        return df.loc[:, [*[col for col in df.columns if col != target], target]]

    def map2idx(self, columns):
        if not put.is_iter_not_str(columns):
            columns = [columns]
        return self.df.columns.get_indexer_for(columns)

    def get_cont_features(self, cont_cols=None, thres='log2'):
        if cont_cols is None:
            if isinstance(thres, str):
                thres = getattr(np, thres)(len(self.df))
            n_uniq = self.df_dense._get_numeric_data().nunique()
            cont_cols = list(n_uniq[n_uniq > thres].index)
            self.cont_cols = cont_cols
        else:
            self.cont_cols = cont_cols

        self.date_cols = self.df.select_dtypes(include=['datetime64', 'timedelta64']).columns
        self.disc_cols = [c for c in self.df.columns
                          if c not in [*self.cont_cols, *self.date_cols, *self.sparse_cols]]
        self.regr = self.target in self.cont_cols

    def encode(self):
        for cat in [*self.disc_cols, *self.sparse_cols]:
            codes, uniques = pd.factorize(self.df[cat].replace(np.nan, 'nan'), sort=True)#, use_na_sentinel=False)
            self.df[cat] = codes
            self.idx2categ[cat] = {k: v for k, v in enumerate(uniques)}
            self.categ_map[cat] = {v: k for k, v in enumerate(uniques)}

        df_date = self.df[self.date_cols]
        df_date_num = df_date.astype(int) / 1e9 / 60 / 60 / 24
        df_date_num[df_date.isna()] = np.nan
        self.df[self.date_cols] = df_date_num
        self.cont_cols = [*self.cont_cols, *self.date_cols]

    def keep_disc_na(self):
        df_keep_disc_na = self.df.copy()
        for col, cate2idx in self.categ_map.items():
            if 'nan' in cate2idx.keys():
                df_keep_disc_na[col] = df_keep_disc_na[col].replace(cate2idx['nan'], np.nan)
        return df_keep_disc_na


    def print_col_info(self):
        print(f"Continuous Columns:")
        print("\n".join([f"\t{s}" for s in self.cont_cols]), "\n")

        print(f"Discrete Columns Encoding:")
        print("\n".join([f"\t{k}: {v}" for k, v in self.categ_map.items()]))

    # -------------------- EDA PLOTS --------------------
    def data_quality_check(self, check_zero=False, miss_thres=0., one_figsize=(15, 5)):
        pplot.data_quality_check(self.df_orig, self.disc_cols, self.sparse_cols,
                                 check_zero, miss_thres, one_figsize=one_figsize)

    def plot_categ_stats(self, **kwargs):
        for categ_col in self.disc_cols:
            pplot.plot_categ_stats(self.df_orig[categ_col], **kwargs)
            plt.show()

    def dist_plot(self, title='Histogram', stat='density', one_figsize=(5, 3), n_col=3, **kwargs):
        layout, figsize = pplot.scale_figsize(self.df.shape[1], one_figsize, n_col)
        fig, ax = plt.subplots(*layout, figsize=figsize)

        for n, c in enumerate(self.df.columns):
            i, j = n // layout[1], n % layout[1]
            one_ax = ax[i, j]
            kde = c in self.cont_cols
            sns.histplot(self.df[c], stat=stat, ax=one_ax, kde=kde, **kwargs)
            one_ax.set_title(c)
            one_ax.set_xlabel(None)

        plt.suptitle(title, y=1)
        plt.tight_layout()
        plt.show()

    def dist_plot_by_categ(self, categ, kind='auto', bw_method='scott', bw_adjust=5,
                           one_figsize=(5, 5), n_col=3, **kwargs):
        assert categ in self.disc_cols
        assert kind in ['auto', 'kde', 'hist', 'ecdf']
        hue = self.df[categ].map(self.idx2categ[categ])
        layout, figsize = pplot.scale_figsize(len(self.cont_cols), one_figsize, n_col)
        fig, ax = plt.subplots(*layout, figsize=figsize)

        for n, c in enumerate(self.cont_cols):
            i, j = n // layout[1], n % layout[1]
            one_ax = ax[i, j]
            legend = n == 0
            title = c
            if kind == 'ecdf':
                g = sns.ecdfplot(self.df, x=c, hue=hue, ax=one_ax, legend=legend, **kwargs)
                title += ' (ECDF)'
            else:
                kde = kind == 'kde'
                if kind == 'auto':
                    bw = gaussian_kde(self.df[c], bw_method=bw_method).cho_cov[0][0]
                    data_gap = self.df[c].sort_values().diff().max()
                    # so that at every x, there are some data points within +- 5 * bw
                    kde = data_gap <= 2 * bw * bw_adjust
                if kde:
                    g = sns.kdeplot(self.df, x=c, hue=hue, common_norm=False,
                                    ax=one_ax, legend=legend, **kwargs)
                    title += ' (KDE)'
                else:
                    g = sns.histplot(self.df, x=c, hue=hue, stat='density',
                                     common_norm=False, ax=one_ax, legend=legend, **kwargs)
                    title += ' (HIST)'
            if legend:
                sns.move_legend(g, 'lower left', bbox_to_anchor=(-0.5, 0), ncol=1)
            one_ax.set_title(title)
            one_ax.set_xlabel(None)
        plt.suptitle(f'Distribution plots for {categ} Categories', y=1)
        plt.tight_layout()
        #pplot.legend_outside(fig=fig, ax=ax)
        plt.show()

    # -------------------- RELATIONSHIP PLOTS --------------------

    def cluster_features(self, corr_method='dcorr', linkage='complete', n_clusters=None,
                         min_asso=None, power=1, corr_kws={}, **kwargs):
        similarity_matrix = self.calc_corr(corr_method, **corr_kws)
        _, distance_matrix = pstats.corr2dist(similarity_matrix, power=power)
        cluster_book = pstats.feature_agglomeration(distance_matrix, n_clusters=n_clusters, min_asso=min_asso,
                          linkage=linkage, power=power, **kwargs)
        return cluster_book

    def calc_corr(self, corr_method='dcorr', **corr_kws):
        if corr_method in self.similarity_matrix:
            return self.similarity_matrix[corr_method]

        df = self.df_keep_disc_na.copy().drop(columns=self.sparse_cols)
        corr_calc_method = corr_method

        if corr_calc_method == 'MI':
            if 'random_state' not in corr_kws:
                corr_kws['random_state'] = self.random_state
            corr = pstats.mi_corr(df, discrete_features=self.map2idx(self.disc_cols), **corr_kws)
        elif corr_calc_method == 'dcorr':
            corr = pstats.distance_correlation(df, disc_features=self.disc_cols, **corr_kws)
        else:
            corr = df.corr(method=corr_method, **corr_kws)
        self.similarity_matrix[corr_method] = corr
        return corr

    def corr_heatmap(self, corr_method='dcorr', partial=False, dist=False, title=None, corr_kws={}, **kwargs):
        corr = self.calc_corr(corr_method=corr_method, **corr_kws)
        if title is None:
            corr_dist = "Distance" if dist else "Correlation"
            title = f"{corr_method} {corr_dist} Heatmap"
        if partial:
            corr = pstats.pcorr(corr)
            title = 'Partial ' + title
        return pplot.plot_corr(corr, dist=dist, title=title, **kwargs)

    def plot_contingency(self, col1, col2, sort=True, corr=False, **kwargs):
        assert col1 in self.disc_cols and col2 in self.disc_cols
        ctg_tab = pd.crosstab(self.df[col1], self.df[col2])
        ctg_tab = ctg_tab.rename_axis(index=col1).rename_axis(columns=col2)
        ctg_tab = ctg_tab.rename(index=self.idx2categ[col1]).rename(columns=self.idx2categ[col2])
        ctg_tab = ctg_tab.rename(index=str).rename(columns=str)

        title = 'Dummy Correlation' if corr else f'Contingency Table'
        title += f' between {col1}, {col2}'
        return pplot.plot_contingency(ctg_tab, corr=corr, title=title, sort=sort, **kwargs)

    def pairplot(self, columns=None, diag_kind='hist', grid_kws=None, **kwargs):
        if grid_kws is None:
            grid_kws = dict(diag_sharey=False)
        if columns is None:
            columns = self.df.columns
        columns = [col for col in columns if col in self.cont_cols]
        g = sns.pairplot(self.df[columns], diag_kind=diag_kind, grid_kws=grid_kws, **kwargs)
        g.fig.suptitle('Pairplot of Continuous Features', y=1)
        plt.tight_layout()
        plt.show()

    def split_data(self, max_batch_size=None, test_size=0.2, q_clip=0.,
                   scaler=False, oversample=False, random_state=None):
        return split_data(self.df, self.target, self.cont_cols, self.regr,
                          max_batch_size, test_size, q_clip, scaler, oversample, random_state)
