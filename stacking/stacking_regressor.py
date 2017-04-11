from sklearn.base import RegressorMixin

from .stacking_estimator import StackingEstimator


class StackingRegressor(StackingEstimator, RegressorMixin):
    def __init__(self,
                 estimators,
                 meta_estimator,
                 cv,
                 use_original_features=False,
                 use_probas=True,
                 average_predictions=False,
                 meta_features_preprocessing=lambda x: x,
                 target_preprocessing=lambda x, y: y,
                 data_preprocessing=None,
                 verbose=0):

        super(StackingRegressor, self).__init__(estimators,
                                                meta_estimator,
                                                cv,
                                                use_original_features,
                                                use_probas,
                                                average_predictions,
                                                meta_features_preprocessing,
                                                target_preprocessing,
                                                data_preprocessing,
                                                verbose)
