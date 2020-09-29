import pandas as pd
from lyft.data import train_data, clean_df
from lyft.functions import haversine_vectorized, minkowski_distance
from lyft.parameters DIST_ARGS
from sklearn.base import BaseEstimator, TransformerMixin


class DateEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, time_column, time_zone_name="America/New_York"):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[["dow", "hour", "month", "year"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class DistanceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, distance_type="haversine"):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "haversine":
            X["distance"] = haversine_vectorized(X, **DIST_ARGS)
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, 1)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, 2)
        return X[["distance"]]

    def fit(self, X, y=None):
        return self
