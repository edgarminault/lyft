import time
import numpy as np


def globe_distance(
    df,
    start_lat="start_lat",
    start_lon="start_lon",
    end_lat="end_lat",
    end_lon="end_lon",
):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Vectorized version of the haversine distance for pandas df
    Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(
        df[start_lon].astype(float)
    )
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(
        df[end_lon].astype(float)
    )
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    distance = 6371 * c
    return distance


def minkowski_distance(
    df,
    p,
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
):
    """
    Calculate 1 to n dimensions distances.
    """
    x1 = df[start_lon]
    x2 = df[end_lon]
    y1 = df[start_lat]
    y2 = df[end_lat]
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)


def compute_error(y_pred, y_true):
    """
    Computes an error hash comprised of both Mean Absolute Error and Mean Squared
    Error.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {"mae": mae, "mse": mse}
