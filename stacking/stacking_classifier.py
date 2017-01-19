from sklearn.base import ClassifierMixin

from .stacking_estimator import StackingEstimator


class StackingClassifier(StackingEstimator, ClassifierMixin):
    def __init__(self,
                 estimators,
                 meta_estimator,
                 cv,
                 use_original_features=False,
                 use_probas=True,
                 average_predictions=False,
                 meta_features_preprocessing=lambda x: x,
                 target_preprocessing=lambda x, y: y,
                 verbose=0):

        super(StackingClassifier, self).__init__(estimators,
                                                 meta_estimator,
                                                 cv,
                                                 use_original_features,
                                                 use_probas,
                                                 average_predictions,
                                                 meta_features_preprocessing,
                                                 target_preprocessing,
                                                 verbose)

    def predict_proba(self, X):
        meta_features = self._predict_meta_features(X)
        meta_features = self._preprocess_meta_features(meta_features)

        return self.meta_est_.predict_proba(meta_features)
