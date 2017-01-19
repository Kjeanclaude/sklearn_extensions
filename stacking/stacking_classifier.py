from sklearn.base import ClassifierMixin

from .stacking_estimator import StackingEstimator


class StackingClassifier(StackingEstimator, ClassifierMixin):

    def __init__(self, *args, **kwargs):
        super(StackingClassifier, self).__init__(*args, **kwargs)

    def predict_proba(self, X):
        meta_features = self._predict_meta_features(X)
        meta_features = self._preprocess_meta_features(meta_features)

        return self.meta_est_.predict_proba(meta_features)
