from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.utils.validation import _is_fitted
from sklearn.preprocessing import QuantileTransformer
from scipy.stats import norm
import numpy as np

class CompositeNB(object):
    def __init__(self, cont_cols=None, alpha=1.0, gaussian=False):
        self.cont_cols = cont_cols
        self.nb_disc = CategoricalNB(alpha=alpha)
        self.nb_cont = GaussianNB()
        self.gaussian = gaussian

    def _split_disc_cont(self, X):
        if self.cont_cols is None:
            cont_cols = range(X.shape[1])
        else:
            cont_cols = self.cont_cols
        X_disc = np.delete(X, cont_cols, axis=1)
        X_cont = X[:, cont_cols]
        assert not np.any(np.nanmin(X_disc, axis=0) < 0), f"Your X_disc should >=0, but {X_disc.min(axis=0)}"
        X_disc += 1  # So that we can have a 'buffer' label 0 representing unseen test labels
        return X_disc, X_cont

    def fit(self, X, y, sample_weight=None):
        X_disc, X_cont = self._split_disc_cont(X)
        self.X_train_means = np.nanmean(X_cont, axis=0)
        X_cont = np.where(np.isnan(X_cont), self.X_train_means, X_cont)

        if self.gaussian:
            self.qt = QuantileTransformer()
            self.qt.set_params(n_quantiles=min(X_cont.shape[0], self.qt.subsample))
            X_cont = self.qt.fit_transform(X_cont)
            X_cont = norm.ppf(X_cont).clip(-3, 3)

        if np.size(X_disc) > 0:
            self.nb_disc.fit(X_disc, y, sample_weight)
            class_count = self.nb_disc.class_count_
        if np.size(X_cont) > 0:
            self.nb_cont.fit(X_cont, y, sample_weight)
            class_count = self.nb_cont.class_count_

        self.class_prior_ = class_count / np.sum(class_count)
        self.X_train_labels = [np.unique(X_disc[:, i]) for i in range(X_disc.shape[1])]

    def predict_proba(self, X):
        X_disc, X_cont = self._split_disc_cont(X)
        X_cont = np.where(np.isnan(X_cont), self.X_train_means, X_cont)

        if self.gaussian:
            X_cont = self.qt.transform(X_cont)
            X_cont = norm.ppf(X_cont).clip(-3, 3)

        is_trained_labels = [np.isin(X_disc[:, i], self.X_train_labels[i]) for i in range(X_disc.shape[1])]
        is_trained_labels = np.array(is_trained_labels).T
        X_disc = np.where(is_trained_labels, X_disc, 0)

        if _is_fitted(self.nb_disc):
            y_pred_prob_disc = self.nb_disc.predict_proba(X_disc)
        else:
            y_pred_prob_disc = self.class_prior_

        if _is_fitted(self.nb_cont):
            y_pred_prob_cont = self.nb_cont.predict_proba(X_cont)
        else:
            y_pred_prob_cont = self.class_prior_

        y_pred_prob = (y_pred_prob_disc * y_pred_prob_cont)
        y_pred_prob = y_pred_prob / self.class_prior_
        y_pred_prob = y_pred_prob / y_pred_prob.sum(axis=1)[:, None]
        return y_pred_prob

    def predict(self, X):
        y_pred_prob = self.predict_proba(X)
        y_pred = np.argmax(y_pred_prob, axis=1)
        return y_pred