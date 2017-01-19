from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone

import numpy as np


class StackingClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 estimators,
                 meta_classifier,
                 cv,
                 use_proba=True,
                 average_proba=False,
                 verbose=0):

        self.estimators = estimators
        self.meta_classifier = meta_classifier
        self.cv = cv
        self.use_proba = use_proba
        self.average_proba = average_proba
        self.verbose = verbose
