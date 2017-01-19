from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone

import numpy as np


class StackingEstimator(BaseEstimator):

    def __init__(self,
                 estimators,
                 meta_estimator,
                 cv,
                 use_original_features=False,
                 use_probas=True,
                 average_predictions=False,
                 meta_features_preprocessing=lambda x: x,
                 target_preprocessing=lambda x, y: y,
                 verbose=2):

        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.use_original_features = use_original_features
        self.use_probas = use_probas
        self.average_predictions = average_predictions
        self.meta_features_preprocessing = meta_features_preprocessing
        self.target_preprocessing = target_preprocessing
        self.verbose = verbose

    def _preprocess_meta_features(self, X):

        return self.meta_features_preprocessing(X)

    def _preprocess_target(self, X, y):

        return self.target_preprocessing(X, y)

    def _predict_meta_features(self, X):
        if self.use_probas:
            probas = np.asarray([est.predict_proba(X)
                                 for est in self.ests_])
            if self.average_predictions:
                vals = np.average(probas, axis=0)
            else:
                vals = np.concatenate(probas, axis=1)
        else:
            vals = np.column_stack([est.predict(X) for est in self.ests_])

            if self.average_predictions:
                vals = np.average(vals, axis=1)

        if self.use_original_features:
            vals = np.hstack((X, vals))

        return vals

    def _fit_estimators(self, X, y):
        self.ests_ = [clone(est) for est in self.estimators]

        if self.verbose > 0:
            print("Fitting {} estimators...".format(len(self.ests_)))

        for i, est in enumerate(self.ests_):
            if self.verbose > 0:
                print("Fitting estimators {est_i}: {est_name} ({est_i}/{est_count})"
                      .format(est_i=i+1, est_name=type(est).__name__.lower(), est_count=len(self.ests_)))

            if self.verbose > 1 and hasattr(est, 'verbose'):
                est.set_params(verbose=self.verbose - 1)

            est.fit(X, y)

    def _fit_meta_features(self, X, y):
        meta_features = None
        for train_index, test_index in self.cv:
            self._fit_estimators(X[train_index], y[train_index])
            local_meta_features = self._predict_meta_features(X[test_index])
            local_meta_features = self._preprocess_meta_features(local_meta_features)

            if meta_features is None:
                meta_features = np.zeros((X.shape[0], local_meta_features.shape[1]))

            meta_features[test_index] = local_meta_features

        return np.concatenate(meta_features)

    def _fit_meta_classifier(self, X, y):
        if self.verbose > 0:
            print("Fitting meta classifier {est_name}"
                  .format(est_name=type(self.meta_est_).__name__.lower()))

        if self.verbose > 1 and hasattr(self.meta_est_, 'verbose'):
            self.meta_est_.set_params(verbose=self.verbose - 1)

        self.meta_est_ = clone(self.meta_estimator)

    def fit(self, X, y):
        meta_features = self._fit_meta_features(X, y)

        meta_y = self._preprocess_target(X, y)
        self._fit_meta_classifier(meta_features, meta_y)

        return self

    def predict(self, X):
        meta_features = self._predict_meta_features(X)
        meta_features = self._preprocess_meta_features(meta_features)

        return self.meta_est_.predict(meta_features)
