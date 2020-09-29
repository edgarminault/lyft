import pandas as pd
from google.cloud import storage

from lyft.parameters import BUCKET_NAME, AWS_PATH, DIST_ARGS


def train_data(nrows=10000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    path = AWS_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def clean_df(df, test=False):
    """Based on EDA notebook findings.
    - outliers in latitude and longitude deleted
    - fare and passengers discrepencies deleted
    """
    df = df.dropna(how="any", axis="rows")
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-75, right=-72)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-75, right=-72)]
    return df


if __name__ == "__main__":
    params = dict(
        nrows=1000,
        local=False,  # set to False to get data from GCP (Storage or BigQuery)
    )
    df = train_data(**params)
