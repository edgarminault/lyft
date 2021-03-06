### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### MLFLOW URI Link - - - - - - - - - - - - - - - - - - - -

MLFLOW_URI = "https://mlflow.lewagon.co/"

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

PROJECT_ID = "wagon-bootcamp-288408"

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = "wagon-ml-edgarminault-gcp"
BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

TEST_DATA_PATH = "data/test.csv"

##### Training  - - - - - - - - - - - - - - - - - - - - - -

AWS_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = "lyft"

# model version folder name (where the trained model.joblib file will be stored)

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### Distance Arguments  - - - - - - - - - - - - - - - - - -

DIST_ARGS = dict(
    start_lat="pickup_latitude",
    start_lon="pickup_longitude",
    end_lat="dropoff_latitude",
    end_lon="dropoff_longitude",
)
