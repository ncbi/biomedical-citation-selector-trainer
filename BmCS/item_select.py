from sklearn.base import BaseEstimator, TransformerMixin

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, column):
        self.column = column

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.column]

    def get_feature_names(self):
        return [self.column]