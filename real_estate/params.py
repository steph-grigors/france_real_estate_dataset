import os

##################  VARIABLES  #####################
DATA_SIZE = os.environ.get("DATA_SIZE")
CHUNK_SIZE = os.environ.get("CHUNK_SIZE")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")


##################  PATHS  #####################
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code","steph-grigors","real_estate_dataset","data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "code","steph-grigors", "real_estate_dataset", "training_outputs")

##################  DATA EXTRACTION PATHS  #####################
NPZ_FILE_PATH = os.path.join(LOCAL_DATA_PATH, "transactions.npz")
OUTPUT_DIR = os.path.join(LOCAL_DATA_PATH, "processed_chunks")
OUTPUT_FILE = os.path.join(LOCAL_DATA_PATH, "full_transactions_dataset", "transactions_combined.csv")
