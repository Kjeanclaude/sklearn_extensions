from sklearn.base import RegressorMixin

from .stacking_estimator import StackingEstimator


class StackingRegressor(StackingEstimator, RegressorMixin):

    def __init__(self, *args, **kwargs):
        super(StackingRegressor, self).__init__(*args, **kwargs)
