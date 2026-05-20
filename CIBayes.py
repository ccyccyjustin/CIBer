from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import _is_fitted
from sklearn.metrics import d2_absolute_error_score, r2_score, log_loss, mean_squared_error, f1_score
from sklearn.preprocessing import QuantileTransformer

from scipy.stats import norm, gaussian_kde
from scipy.integrate import simpson

from KDEpy import FFTKDE
from KDEpy.utils import cartesian

from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda import FDataGrid

import plot as pplot
import time


def get_dummies(df, disc_cols=None, df_dtype='float32', **kwargs):
    # If disc_cols is None, the original logic encodes all columns.
    # Otherwise, it only encodes the specified list.
    target_cols = disc_cols if disc_cols is not None else df.columns.tolist()

    df2 = pd.get_dummies(df, columns=target_cols, dtype=df_dtype, sparse=True, **kwargs)
    assert df2.shape[1] == df[target_cols].apply(lambda s: s.nunique()).sum()
    return df2

class MCA(object):
    def __init__(self, expl_var=1.0, alpha=1e-6, tol=1e-10):
        self.unexpl_thres = 1.0 - expl_var + tol # 1.0 - 1.0 + 1e-8
        self.alpha = alpha

        self._eigenvalues = None
        self.total_inertia_ = None
        self.explained_variance_ratio_ = None
        self.row_coordinates_ = None
        self.col_coordinates_ = None
        self.n_components = None
        self._decimals = np.log10(1 / self.alpha).astype(int) - 1

    def fit(self, df):
        # 1. Indicator Matrix (Z)
        # One-hot encoding the categorical features
        t0 = time.time()
        df2 = get_dummies(df) # ~0.2s
        t1 = time.time()
        print(t1-t0)
        Z = df2.values
        m, N = Z.shape
        n = df.shape[1]  # Number of original categorical features

        # 3. Compute the Burt-style Kernel: (m * P.T @ P)
        Burt = Z.T @ Z #0.5s
        t1 = time.time()
        print(t1-t0)

        # 2. Compute Column Marginals
        col_sums = np.diag(Burt)
        c = col_sums / (m * n)
        w = 1.0 / np.sqrt(c)

        term1 = Burt / (m * n ** 2)
        term2 = np.outer(c, c)
        Centered_Kernel = term1 - term2
        K = (Centered_Kernel * w[:, None]) * w[None, :]

        # 4. Eigen-decomposition (EVD)
        # Using eigh because K is symmetric. Returns sorted eigenvalues.
        ramp = np.linspace(0, self.alpha, N)
        K = K + np.diag(ramp)
        self.K = K
        evals, V = np.linalg.eigh(K)  # 0.3s
        t1 = time.time()
        print(t1-t0)

        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        self._eigenvalues = evals[idx]
        self.V_ = V[:, idx]

        # 7. Benzécri Correction
        self._apply_benzecri(n)

        # 8. Coordinates
        # Row coordinates (Factor Scores)
        # self.row_coordinates_ = (1/np.sqrt(r[:, None])) * U_svd[:, :self.n_components] * Sigma[:self.n_components]
        col_coordinates_ = w[:, None] * self.V_[:, :self.n_components] * np.sqrt(self._eigenvalues[:self.n_components])
        self.col_coordinates_ = pd.DataFrame(col_coordinates_, index=df2.columns)
        return

    def _apply_benzecri(self, n):
        """
        Calculates corrected eigenvalues and explained variance ratio.
        n: number of original categorical variables.
        """
        # Benzécri formula for eigenvalues > 1/n
        threshold = 1.0 / n
        corrected_lambdas = np.where(
            self._eigenvalues > threshold + self.alpha,
            ((n / (n - 1)) * (self._eigenvalues - threshold)) ** 2,
            0
        )

        # Only dimensions with lambda > 1/n contribute to the corrected total inertia
        total_inertia_corrected_ = np.sum(corrected_lambdas)
        explained_variance_ratio_ = corrected_lambdas / total_inertia_corrected_
        #uniq_eigval = np.diff(self.eigenvalues_, prepend=np.inf) < -1e-6
        next_change = np.diff(self._eigenvalues, append=-np.inf) < -self.alpha
        large_ratio = explained_variance_ratio_.cumsum() >= 1 - self.unexpl_thres
        self.n_components = np.where(large_ratio & next_change)[0][0] + 1
        self.explained_variance_ratio_ = explained_variance_ratio_[:self.n_components]

        self.eigenvalues = self._eigenvalues.round(self._decimals)
        values, counts = np.unique(self.eigenvalues, return_counts=True, sorted=True)
        self.eigval_multiplicity = pd.Series(counts, index=values)[::-1]

    def transform(self, df):
        Z = get_dummies(df).values
        Z_rot = (Z @ self.col_coordinates_.astype('float32')) / df.shape[1]
        return Z_rot

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_mapping(self, df, sort=False):
        t0 = time.time()
        df_rot = self.transform(df)
        t1= time.time()
        print(t1-t0)

        index = list(map(lambda s: '/'.join(s), df.values.astype(str)))
        columns = [f'{c} ({self.explained_variance_ratio_[c]:.2%})' for c in df_rot.columns]
        mapping = pd.DataFrame(df_rot.values, index=index, columns=columns).drop_duplicates()
        if sort:
            mapping = mapping.sort_values(list(mapping.columns))
        t1 = time.time()
        print(t1-t0)
        return mapping

    def plot_mapping(self, df, sort=True, col_norm=False, figsize=(20,10), first_n=None, **kwargs):
        mapping = self.get_mapping(df, sort)
        if col_norm:
            mapping = mapping / mapping.abs().max()
        if first_n is not None:
            mapping = mapping.iloc[:, :first_n]
        pplot.heatmap(mapping, center=0, title=f'Mapped Values of {list(df.columns)}',
                      figsize=figsize, annot=False)


class BaseCIB(ABC):
    def __init__(self, cont_cols=None, z_clip=5, criterion='mode',
            max_features=30, tol=1e-4, n_grid=300, margin=0.01):
        assert criterion in ['mean', 'median', 'mode']
        self.cont_cols = cont_cols
        self.cont_cols_ = None
        self.disc_cols_ = None
        self.z_clip = z_clip
        self.criterion = criterion
        self.max_features = max_features
        self.tol = tol
        self.n_grid = n_grid
        self.margin = margin

        self.X_transformer = QuantileTransformer()
        self.y_transformer = QuantileTransformer()
        self.kde_results = {}
        self.tables = {}
        self.conditionals = {}
        self.prior = None
        self.grid_1d = None
        self.features = None
        self.y_train_orig = None

    def _split_disc_cont(self, X):
        if self.cont_cols is None:
            self.cont_cols_ = range(X.shape[1])
        else:
            self.cont_cols_ = self.cont_cols
        self.disc_cols_ = [c for c in range(X.shape[1]) if c not in self.cont_cols_]

        X_disc = np.delete(X, self.cont_cols_, axis=1)
        X_cont = X[:, self.cont_cols_]
        assert not np.any(np.nanmin(X_disc, axis=0) < 0), f"Your X_disc should >=0, but {X_disc.min(axis=0)}"
        X_disc += 1 # So that we can have a 'buffer' label 0 representing unseen test labels
        return X_disc, X_cont

    def _to_gaussian(self, data, transformer):
        if not _is_fitted(transformer):
            transformer.set_params(n_quantiles=min(data.shape[0], transformer.subsample))
            transformer.fit(data)

        data2 = transformer.transform(data)
        data2 = norm.ppf(data2).clip(-self.z_clip, self.z_clip)
        return data2

    def _get_1d_gauss_grid(self, n_grid, margin):
        grid_max = self.z_clip * (1 + margin)
        grid = np.linspace(-grid_max, grid_max, n_grid)
        return grid

    def _calc_density(self, data, grid_1d):
        n_grid = len(grid_1d)
        if np.ndim(data) == 1:
            n, d = len(data), 1
        else:
            n, d = data.shape
        bw = (4 / (d + 2) / n) ** (1 / (d + 4))

        kernel = FFTKDE(bw=bw)
        kernel.fit(data)

        grid = cartesian([grid_1d] * d)
        Z = kernel.evaluate(grid)

        # So that, e.g. in 2d, grid[:, :, 0] = X, grid[:, :, 1] = Y
        grid = grid.reshape((*(n_grid, ) * d, d))
        Z = Z.reshape((n_grid, ) * d)
        return {'grid': grid, 'density': Z, 'kernel': kernel}

    @abstractmethod
    def _preprocess_y(self, y_train):
        pass

    def _preprocess_data(self, X_train, y_train=None):
        X_disc, X_cont = self._split_disc_cont(X_train)
        X_cont = self._to_gaussian(X_cont, self.X_transformer)

        if y_train is not None:
            assert np.isfinite(y_train).all(), "Does not accept nan on y!"
            y_train = self._preprocess_y(y_train)
        return X_disc, X_cont, y_train

    def inverse_transform_y(self, y_gauss):
        return self.y_transformer.inverse_transform(norm.cdf(y_gauss).reshape(-1, 1)).flatten()

    @staticmethod
    def _reindex_to_orig(x_orig, y, idx=None):
        if idx is None:
            idx = np.isfinite(x_orig)
        y_orig = np.full((x_orig.shape[0], y.shape[1]), np.nan)
        y_orig[idx] = y
        return y_orig

    @abstractmethod
    def fit_prior(self, y_train):
        pass

    @property
    @abstractmethod
    def prior_prob(self):
        pass

    @abstractmethod
    def normalize_posterior(self, posterior):
        pass

    def update_posterior(self, prior, conditionals, var_smoothing=1e-30):
        if not isinstance(conditionals, list):
            conditionals = [conditionals]

        log_posterior = np.log(prior)

        for cond in conditionals:
            log_cond = np.log(np.clip(cond, a_min=var_smoothing, a_max=None))
            log_cond[~np.isfinite(cond).any(axis=1)] = 0
            log_posterior = log_posterior + log_cond

        posterior = np.exp(log_posterior)
        posterior = self.normalize_posterior(posterior)
        return posterior

    def calc_post_mean(self, posterior):
        posterior_mean = simpson(posterior * self.prior['grid'], self.prior['grid'])
        data_mean = self.y_train_orig.mean()
        posterior_mean = posterior_mean * data_mean / np.nanmean(posterior_mean)
        return posterior_mean

    def calc_post_median(self, posterior):
        cum_prob = np.cumsum(posterior, axis=1)
        post_median = np.argmax(cum_prob >= 0.5, axis=1)
        post_median = self.prior['grid'][post_median]
        return post_median

    def calc_post_mode(self, posterior):
        posterior_mode = np.argmax(posterior, axis=1) #TODO: map back to the prior labels...
        posterior_mode = self.prior['class'][posterior_mode]
        return posterior_mode

    def post_estimate(self, posterior, criterion):
        FUNCS = {'mean': self.calc_post_mean,
                 'median': self.calc_post_median,
                 'mode': self.calc_post_mode}
        return FUNCS[criterion](posterior)

    @abstractmethod
    def calc_score(self, y_train, post, criterion):
        pass

    def _feature_selection(self, conditionals: dict, criterion, max_features, tol):
        """
        Performs Forward Selection based on training loss
        :param conditionals:
        :param criterion:
        :param max_features:
        :param tol:
        :return:
        """
        prior = self.prior_prob[None, :]
        score_curve = {None: 1e-6}

        for i in range(max_features):
            score_i = {}
            posterior_i = {}
            for col in sorted(conditionals):
                if col in score_curve:
                    continue
                cond_density = conditionals[col]
                post = self.update_posterior(prior, cond_density)
                score = self.calc_score(self.y_train_orig, post, criterion) #TODO: AIC
                score_i[col] = score
                posterior_i[col] = post

            best_col_i = pd.Series(score_i).idxmax()
            best_score_i = score_i[best_col_i]
            pct_score_incr = (best_score_i / list(score_curve.values())[-1]) - 1
            if pct_score_incr < tol:
                break

            score_curve[best_col_i] = best_score_i
            prior = posterior_i[best_col_i].copy() # Prior will be updated as posterior after each iter

        score_curve.pop(None)
        score_curve = pd.Series(score_curve).rename(int)
        return {'posterior': prior, 'fit_score': score_curve, 'conditionals': conditionals}

    @abstractmethod
    def calc_cont_cond(self, x, y_train, has_data):
        pass

    @abstractmethod
    def calc_disc_cond(self, x, y_train, n_grid):
        pass

    def fit(self, X_train, y_train, sample_weight=None):
        n, p = X_train.shape
        if self.max_features is not None:
            max_features = min(self.max_features, p)
        else:
            max_features = None
        self.y_train_orig = y_train.copy()

        X_disc, X_cont, y_train = self._preprocess_data(X_train, y_train)
        self.grid_1d = self._get_1d_gauss_grid(self.n_grid, self.margin)
        self.prior = self.fit_prior(y_train)

        data_conditionals = {}
        # Calculate the cond prob (U_i | V), the transformed Gaussians
        for c in range(X_cont.shape[1]):
            col_name = self.cont_cols_[c]
            x = X_cont[:, c]
            has_data = np.isfinite(x)
            x_idx = np.searchsorted(self.grid_1d, x[has_data])

            kde_result, cond_density = self.calc_cont_cond(x, y_train, has_data)
            self.kde_results[c] = kde_result
            self.conditionals[col_name] = cond_density

            data_cond_density = cond_density[x_idx]
            data_cond_density = self._reindex_to_orig(x, data_cond_density)
            data_conditionals[col_name] = data_cond_density

        for d in range(X_disc.shape[1]):
            col_name = self.disc_cols_[d]
            x = X_disc[:, d]
            has_data = np.isfinite(x)
            x_idx = x[has_data]

            table = self.calc_disc_cond(x, y_train, self.n_grid)
            self.conditionals[col_name] = table

            data_cond_prob = table.loc[x_idx]
            data_cond_prob = self._reindex_to_orig(x, data_cond_prob)
            data_conditionals[col_name] = data_cond_prob

        if max_features is not None:
            final_fit_res = self._feature_selection(data_conditionals, criterion=self.criterion,
                                                    max_features=max_features, tol=self.tol)
            posterior = final_fit_res['posterior']
            self.features = list(final_fit_res['fit_score'].index)
        else:
            self.features = list(data_conditionals.keys())
            posterior = self.update_posterior(self.prior_prob[None, :], list(data_conditionals.values()))
            final_fit_res = {'posterior': posterior, 'conditionals': data_conditionals}

        posterior_mean = self.post_estimate(posterior, criterion=self.criterion)
        return posterior_mean, final_fit_res

    def predict_proba(self, X_test):
        X_disc, X_cont, _ = self._preprocess_data(X_test, y_train=None)
        data_conditionals = {}

        for c in range(X_cont.shape[1]):
            col_name = self.cont_cols_[c]
            if col_name not in self.features:
                continue
            x = X_cont[:, c]
            has_data = np.isfinite(x)
            x_idx = np.searchsorted(self.grid_1d, x[has_data])

            data_cond_density = self.conditionals[col_name][x_idx]
            data_cond_density = self._reindex_to_orig(x, data_cond_density)
            data_conditionals[col_name] = data_cond_density

        for d in range(X_disc.shape[1]):
            col_name = self.disc_cols_[d]
            if col_name not in self.features:
                continue
            x = X_disc[:, d]

            data_cond_prob = self.conditionals[col_name]
            seen_label = np.isin(x, data_cond_prob.index)
            has_data = np.isfinite(x) & seen_label

            data_cond_prob = data_cond_prob.loc[x[has_data]]
            data_cond_prob = self._reindex_to_orig(x, data_cond_prob, idx=has_data)
            data_conditionals[col_name] = data_cond_prob

        posterior = self.update_posterior(self.prior_prob[None, :], list(data_conditionals.values()))
        return posterior#, data_conditionals

    def predict(self, X_test, criterion='mode'):
        posterior = self.predict_proba(X_test)
        posterior_mean = self.post_estimate(posterior, criterion)
        return posterior_mean

    def plot_joint_density(self, X, y, columns=None, figsize=(10, 10)):
        _, X_cont, y = self._preprocess_data(X, y)
        assert len(self.kde_results) > 0, "Not fitted!"
        if columns is None:
            columns = list(self.kde_results.keys())
        if isinstance(columns, int):
            columns = [columns]

        for c in columns:
            Z = self.kde_results[c]['density']
            fig, ax = plt.subplots(figsize=figsize)
            extent = [-self.z_clip, self.z_clip, -self.z_clip, self.z_clip]
            ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=extent, aspect='auto')
            ax.plot(X_cont[:, c], y, 'k.', markersize=2)
            plt.show()

    def plot_cond_prob(self, columns=None):
        if columns is None:
            columns = self.disc_cols_
        for d in columns:
            if d not in self.disc_cols_:
                continue
            tab = self.conditionals[d]
            tab2 = tab.replace(0, np.nan).dropna(how='all').T
            tab2 = tab2.rename(columns=lambda x: f'{x - 1}' if x > 0 else 'Missing')
            tab2.pplot(kind='area', title=f'Cond Prob of Column {d}')


class CIBClassifier(BaseCIB):
    def __init__(self, cont_cols=None, z_clip=5, criterion='mode', alpha=1.,
            max_features=30, tol=1e-4, n_grid=300, margin=0.01):
        super().__init__(cont_cols, z_clip, criterion, max_features, tol, n_grid, margin)
        self.alpha = alpha

    def _preprocess_y(self, y_train):
        return y_train

    def fit_prior(self, y_train):
        y_labels = np.unique(y_train)
        prob = np.array([(y_train == c).sum() for c in y_labels])
        prob = prob / prob.sum()
        prior = {'class': y_labels, 'prob': prob}
        return prior

    @property
    def prior_prob(self):
        return self.prior['prob']

    def normalize_posterior(self, posterior):
        return posterior / posterior.sum(axis=1)[:, None]

    def fit_left_cdf(self, cdf, q1, q2):
        lb = np.argmin(cdf <= q1, axis=0)
        lb2 = np.argmin(cdf <= q2, axis=0)
        sig_l = (self.grid_1d[lb2] - self.grid_1d[lb]) / (norm.ppf(q2) - norm.ppf(q1))
        assert (sig_l >= 0).all(), f"lb1:{lb}, lb2:{lb2}"
        mu_l = np.where(sig_l > 1e-4, self.grid_1d[lb] - sig_l * norm.ppf(q1), 0)
        return mu_l, sig_l

    def fit_right_cdf(self, cdf, q1, q2):
        ub = np.argmax(cdf >= 1 - q1, axis=0)
        ub2 = np.argmax(cdf >= 1 - q2, axis=0)
        sig_r = (self.grid_1d[ub] - self.grid_1d[ub2]) / (norm.ppf(1 - q1) - norm.ppf(1 - q2))
        assert (sig_r >= 0).all(), f"ub1:{ub}, ub2:{ub2}"
        mu_r = np.where(sig_r > 1e-4, self.grid_1d[ub] - sig_r * norm.ppf(1 - q1), 0)
        return mu_r, sig_r

    def clip_density(self, pdf, q1=0.01, q2=0.05):
        step_size = np.diff(self.grid_1d).mean()
        pdf = pdf / pdf.sum(axis=0) / step_size
        cdf = pdf.cumsum(axis=0) * step_size

        mu_l, sig_l = self.fit_left_cdf(cdf, q1, q2)
        mu_r, sig_r = self.fit_right_cdf(cdf, q1, q2)

        z_left = np.where(sig_l > 0, np.divide(self.grid_1d[:, None] - mu_l, sig_l, where=sig_l > 0), np.nan)
        z_right = np.where(sig_r > 0, np.divide(self.grid_1d[:, None] - mu_r, sig_r, where=sig_r > 0), np.nan)

        gauss_left = np.where(np.isfinite(z_left), norm.pdf(z_left), 0)
        gauss_right = np.where(np.isfinite(z_right), norm.pdf(z_right), 0)

        pdf = np.where(cdf < q1, gauss_left, pdf)
        pdf = np.where(cdf > 1-q1, gauss_right, pdf)
        pdf = pdf / pdf.sum(axis=0) / step_size
        return pdf

    def calc_cont_cond(self, x, y_train, has_data, min_pct=0.01):
        kde_result = {}
        cond_density = []
        for y_class in self.prior['class']:
            x_class = x[has_data & (y_train == y_class)]
            if len(x_class) < min_pct * len(y_train):
                density = norm.pdf(self.grid_1d) # Use density unconditional of class
                kde_result_class = {'density': density}
            else:
                kde_result_class = self._calc_density(x_class, self.grid_1d)  # Already conditional density
            kde_result[y_class] = kde_result_class
            cond_density.append(kde_result_class['density'])
        cond_density = np.array(cond_density).T
        cond_density = self.clip_density(cond_density)
        return kde_result, cond_density

    def calc_disc_cond(self, x, y_train, n_grid):
        table = pd.crosstab(x, y_train) #TODO: to numpy
        table += self.alpha
        table = table / table.sum()
        return table

    def calc_score(self, y_train, posterior, criterion):
        y_pred = self.post_estimate(posterior, criterion)
        return f1_score(y_train, y_pred, average='macro')


class CIBRegressor(BaseCIB):
    def __init__(self, cont_cols=None, z_clip=5, criterion='mean',
            max_features=30, tol=1e-4, n_grid=300, margin=0.01):
        super().__init__(cont_cols, z_clip, criterion, max_features, tol, n_grid, margin)

    def _preprocess_y(self, y_train):
        return self._to_gaussian(y_train.reshape(-1, 1), self.y_transformer).flatten()

    def _fit_prior_kernel(self, y_train_gauss, grid_1d):
        prior_grid = self.inverse_transform_y(grid_1d)
        y_train_orig = self.inverse_transform_y(y_train_gauss)
        prior_kernel = gaussian_kde(y_train_orig, bw_method='silverman')
        prior_density = prior_kernel.evaluate(prior_grid)
        return {'grid': prior_grid, 'density': prior_density, 'kernel': prior_kernel}

    def fit_prior(self, y_train):
        prior = self._fit_prior_kernel(y_train, self.grid_1d)
        return prior

    @property
    def prior_prob(self):
        return self.prior['density']

    def normalize_posterior(self, posterior):
        return posterior / simpson(posterior, self.prior['grid'])[:, None]

    def calc_cont_cond(self, x, y_train, has_data):
        xy = np.column_stack([x, y_train])
        xy = xy[has_data]
        kde_result = self._calc_density(xy, self.grid_1d)
        joint_density = kde_result['density']
        cond_density = (joint_density / norm.pdf(self.grid_1d)[None, :])  # Divide joint by gaussian marginal
        return kde_result, cond_density

    @staticmethod
    def _one_hot(x: np.array):
        x2 = np.nan_to_num(x, nan=0.).astype(int)
        num_classes = np.max(x2) + 1
        num_samples = x.shape[0]

        x_one_hot = np.zeros((num_samples, num_classes)) #TODO: make shape[1] equal to nunique...
        x_one_hot[np.arange(num_samples), x2] = 1
        return x_one_hot

    def compute_disc_cond_prob(self, x_train, y_train, n_grid):
        d = len(np.unique(x_train))
        n = len(x_train)
        bw = (4 / (d + 2) / n) ** (1 / (d + 4))
        kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=bw)

        order = np.argsort(y_train)
        x_train_one_hot = self._one_hot(x_train) # P(X=x | y) = E(one_hot | y) = Kernel_smooth(one_hot...)
        fd = FDataGrid(grid_points=y_train[order],
                       data_matrix=x_train_one_hot[order].T)

        grid_trunc = self._get_1d_gauss_grid(n_grid, margin=0.)
        ks = KernelSmoother(kernel_estimator=kernel_estimator,
                            output_points=grid_trunc)

        ks_res = ks.fit_transform(fd)
        cond_prob = pd.DataFrame(ks_res.data_matrix.T[0], index=ks_res.grid_points[0]) #TODO: np
        return ks, cond_prob

    def calc_disc_cond(self, x, y_train, n_grid):
        _, table = self.compute_disc_cond_prob(x, y_train, n_grid)
        table = table.reindex(self.grid_1d, method='nearest').T
        return table

    def calc_score(self, y_train, posterior, criterion):
        y_pred = self.post_estimate(posterior, criterion)
        return r2_score(y_train, y_pred)

    def evaluate_fit(self, y_true, posterior_mean, rank=False):
        goodness_of_fit = pd.DataFrame({'true': y_true,
                                        'pred': posterior_mean})
        pcorr = goodness_of_fit.corr().loc['true', 'pred']
        scorr = goodness_of_fit.corr('spearman').loc['true', 'pred']
        r2 = r2_score(y_true, posterior_mean)

        if rank:
            goodness_of_fit.rank(pct=True).plot(kind='scatter', x='true', y='pred')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
        else:
            goodness_of_fit.plot(kind='scatter', x='true', y='pred')
            plt.ylim(y_true.min(), y_true.max())
        plt.title(f'Pearson Corr: {pcorr:.2f}, Spearman Corr: {scorr:.2f}, R2: {r2:.4f}')