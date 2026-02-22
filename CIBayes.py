import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.validation import _is_fitted
from sklearn.metrics import d2_absolute_error_score, r2_score, log_loss, mean_squared_error
from sklearn.preprocessing import QuantileTransformer

from scipy.stats import norm, gaussian_kde
from scipy.integrate import simpson

from KDEpy import FFTKDE
from KDEpy.utils import cartesian

from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda import FDataGrid


class GaussTransKDE(object):
    def __init__(self, regr=True, cont_cols=None, z_clip=5):
        self.regr = regr
        self.cont_cols = cont_cols
        self.cont_cols_ = None
        self.disc_cols_ = None
        self.z_clip = z_clip

        self.X_transformer = QuantileTransformer()
        self.y_transformer = QuantileTransformer()
        self.kde_results = {}
        self.tables = {}
        self.conditionals = {}
        self.prior = None
        self.grid_1d = None
        self.features = None

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
            transformer.set_params(n_quantiles=data.shape[0])
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

    def _preprocess_data(self, X_train, y_train=None):
        X_disc, X_cont = self._split_disc_cont(X_train)
        X_cont = self._to_gaussian(X_cont, self.X_transformer)

        if y_train is not None:
            assert np.isfinite(y_train).all(), "Does not accept nan on y!"
            if self.regr:
                y_train = self._to_gaussian(y_train.reshape(-1, 1), self.y_transformer).flatten()
        return X_disc, X_cont, y_train

    @staticmethod
    def _reindex_to_orig(x_orig, y, idx=None):
        if idx is None:
            idx = np.isfinite(x_orig)
        y_orig = np.full((x_orig.shape[0], y.shape[1]), np.nan)
        y_orig[idx] = y
        return y_orig

    def inverse_transform_y(self, y_gauss):
        return self.y_transformer.inverse_transform(norm.cdf(y_gauss).reshape(-1, 1)).flatten()

    def _fit_prior_kernel(self, y_train_gauss, grid_1d):
        prior_grid = self.inverse_transform_y(grid_1d)
        y_train_orig = self.inverse_transform_y(y_train_gauss)
        prior_kernel = gaussian_kde(y_train_orig, bw_method='silverman')
        prior_density = prior_kernel.evaluate(prior_grid)
        return {'grid': prior_grid, 'density': prior_density, 'kernel': prior_kernel}

    def fit_prior(self, y_train):
        if self.regr:
            prior = self._fit_prior_kernel(y_train, self.grid_1d)
        else:
            y_labels = np.unique(y_train)
            prob = np.array([(y_train == c).sum() for c in y_labels])
            prob = prob / prob.sum()
            prior = {'class': y_labels, 'prob': prob}
        return prior

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
        x_train_one_hot = self._one_hot(x_train)
        fd = FDataGrid(grid_points=y_train[order],
                       data_matrix=x_train_one_hot[order].T)

        grid_trunc = self._get_1d_gauss_grid(n_grid, margin=0.)
        ks = KernelSmoother(kernel_estimator=kernel_estimator,
                            output_points=grid_trunc)

        ks_res = ks.fit_transform(fd)
        cond_prob = pd.DataFrame(ks_res.data_matrix.T[0], index=ks_res.grid_points[0])
        return ks, cond_prob

    def update_posterior(self, prior, conditionals, var_smoothing=1e-30):
        if not isinstance(conditionals, list):
            conditionals = [conditionals]

        log_posterior = np.log(prior)

        for cond in conditionals:
            log_cond = np.log(np.clip(cond, a_min=var_smoothing, a_max=None))
            log_cond[~np.isfinite(cond).any(axis=1)] = 0
            log_posterior = log_posterior + log_cond

        posterior = np.exp(log_posterior)
        if self.regr:
            posterior = posterior / simpson(posterior, self.prior['grid'])[:, None]
        else:
            posterior = posterior / posterior.sum(axis=1)[:, None]
        return posterior

    def calc_post_mean(self, posterior, data_mean=None):
        assert self.regr
        posterior_mean = simpson(posterior * self.prior['grid'], self.prior['grid'])
        if data_mean is not None:
            posterior_mean = posterior_mean * data_mean / np.nanmean(posterior_mean)
        return posterior_mean

    def calc_post_median(self, posterior):
        assert self.regr
        cum_prob = np.cumsum(posterior, axis=1)
        post_median = np.argmax(cum_prob >= 0.5, axis=1)
        post_median = self.prior['grid'][post_median]
        return post_median

    def calc_post_mode(self, posterior):
        assert not self.regr
        posterior_mode = np.argmax(posterior, axis=1) #TODO: map back to the prior labels...
        posterior_mode = self.prior['class'][posterior_mode]
        return posterior_mode

    def post_estimate(self, posterior, data_mean=None):
        if self.regr:
            return self.calc_post_mean(posterior, data_mean)
        else:
            return self.calc_post_mode(posterior)

    def _feature_selection(self, conditionals: dict, y_train, max_features, tol):
        if self.regr:
            prior = self.prior['density']
        else:
            prior = self.prior['prob']

        prior = prior[None, :]
        loss_curve = {None: np.inf}
        LOSS_FUNC = mean_squared_error if self.regr else log_loss

        for i in range(max_features):
            loss_i = {}
            posterior_i = {}
            for col in sorted(conditionals):
                if col in loss_curve:
                    continue
                cond_density = conditionals[col]
                post = self.update_posterior(prior, cond_density)
                if self.regr:
                    y_pred = self.post_estimate(post, y_train.mean())
                else:
                    y_pred = post # log loss on the probabilities

                loss_i[col] = LOSS_FUNC(y_train, y_pred)
                posterior_i[col] = post

            best_col_i = pd.Series(loss_i).idxmin()
            best_lost_i = loss_i[best_col_i]
            pct_loss_decr = ((best_lost_i / list(loss_curve.values())[-1]) - 1) * (-1)
            if pct_loss_decr < tol:
                break

            loss_curve[best_col_i] = best_lost_i
            prior = posterior_i[best_col_i].copy() # Prior will be updated as posterior after each iter

        loss_curve.pop(None)
        loss_curve = pd.Series(loss_curve).rename(int)
        return {'posterior': prior, 'fit_loss': loss_curve}

    def fit(self, X_train, y_train, sample_weight=None,
            max_features=30, tol=1e-4, n_grid=300, margin=0.01):
        n, p = X_train.shape
        max_features = min(max_features, p)
        y_train_orig = y_train.copy()

        self.data_mean = y_train_orig.mean()
        X_disc, X_cont, y_train = self._preprocess_data(X_train, y_train)
        self.grid_1d = self._get_1d_gauss_grid(n_grid, margin)
        self.prior = self.fit_prior(y_train)

        data_conditionals = {}
        # Calculate the cond prob (U_i | V), the transformed Gaussians
        for c in range(X_cont.shape[1]):
            col_name = self.cont_cols_[c]
            x = X_cont[:, c]
            has_data = np.isfinite(x)
            x_idx = np.searchsorted(self.grid_1d, x[has_data])

            if self.regr:
                xy = np.column_stack([x, y_train])
                xy = xy[has_data]
                kde_result = self._calc_density(xy, self.grid_1d)
                joint_density = kde_result['density']
                cond_density = (joint_density / norm.pdf(self.grid_1d)[None, :])  # Divide joint by gaussian marginal
            else:
                kde_result = {}
                cond_density = []
                for y_class in self.prior['class']:
                    x_class = x[has_data & (y_train == y_class)]
                    kde_result_class = self._calc_density(x_class, self.grid_1d) # Already conditional density
                    kde_result[y_class] = kde_result_class
                    cond_density.append(kde_result_class['density'])
                cond_density = np.array(cond_density).T

            self.kde_results[c] = kde_result
            self.conditionals[col_name] = cond_density

            data_cond_density = cond_density[x_idx]
            data_cond_density = self._reindex_to_orig(x, data_cond_density)
            data_conditionals[col_name] = data_cond_density

        for d in range(X_disc.shape[1]):
            col_name = self.disc_cols_[d]
            x = X_disc[:, d]
            has_data = np.isfinite(x)

            if self.regr:
                _, table = self.compute_disc_cond_prob(x, y_train, n_grid)
                table = table.reindex(self.grid_1d, method='nearest').T
            else:
                table = pd.crosstab(x, y_train)
                table = table / table.sum()

            self.conditionals[col_name] = table
            data_cond_prob = table.loc[x[has_data]]
            data_cond_prob = self._reindex_to_orig(x, data_cond_prob)
            data_conditionals[col_name] = data_cond_prob

        final_fit_res = self._feature_selection(data_conditionals, y_train_orig, max_features=max_features, tol=tol)
        posterior_mean = self.post_estimate(final_fit_res['posterior'], y_train_orig.mean())
        self.features = list(final_fit_res['fit_loss'].index)
        return posterior_mean, final_fit_res

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

        if self.regr:
            prior = self.prior['density']
        else:
            prior = self.prior['prob']

        posterior = self.update_posterior(prior[None, :], list(data_conditionals.values()))
        return posterior

    def predict(self, X_test):
        posterior = self.predict_proba(X_test)
        posterior_mean = self.post_estimate(posterior, self.data_mean)
        return posterior_mean