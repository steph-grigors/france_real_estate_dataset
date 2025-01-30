import os

##################  VARIABLES  #####################
DATA_SIZE = os.environ.get("DATA_SIZE")
DATA_EXTRACTION_CHUNK_SIZE  = int(os.environ.get("DATA_EXTRACTION_CHUNK_SIZE"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MODEL_TYPE = os.environ.get("MODEL_TYPE")
# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
# MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
# MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
# PREFECT_FLOW_NAME = os.environ.get("PREFECT_FLOW_NAME")
# PREFECT_LOG_LEVEL = os.environ.get("PREFECT_LOG_LEVEL")


##################  CONSTANT PATHS  #####################
LOCAL_PROJECT_PATH = os.path.join(os.path.expanduser('~'), "code","steph-grigors","real_estate_dataset")
LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), "code","steph-grigors","real_estate_dataset","data")
LOCAL_REGISTRY_PATH = os.path.join(os.path.expanduser('~'), "code","steph-grigors", "real_estate_dataset", "training_outputs")

##################  DATA EXTRACTION & PROCESSING PATHS  #####################
NPZ_FILE_PATH = os.path.join(LOCAL_DATA_PATH, "transactions.npz")
RAW_DATASET_CHUNKS_DIR = os.path.join(LOCAL_DATA_PATH, "raw_dataset", "raw_dataset_chunks")
RAW_DATASET_OUTPUT_FILE = os.path.join(LOCAL_DATA_PATH, "raw_dataset", "raw_dataset_full", "transactions.csv")
CLEANED_DATASET_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset", "cleaned.csv")
MERGED_DATASET_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset", "merged.csv")
MERGED_TRAIN_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset","train_set", "train.csv")
MERGED_TEST_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset", "test_set", "test.csv")
PREPROCESSED_TRAIN_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset", "train_set", "train_preprocessed.csv")
PREPROCESSED_TEST_FILE = os.path.join(LOCAL_DATA_PATH, "processed_dataset", "test_set", "test_preprocessed.csv")


##################  CITY MAPPING PATH   #####################
CITY_MAPPING_PATH = os.path.join(LOCAL_PROJECT_PATH, "city_mapping.csv")

DTYPES_RAW = {
    "prix": "float64",
    "departement": "int16",
    "id_ville": "int16",
    "ville": "string",
    "type_batiment": "string",
    "n_pieces": "int16",
    "surface_habitable": "float32",
    "surface_terrains_sols": "float32"
}

DTYPES_CLEANED = {
    "prix": "float64",
    "departement": "int16",
    "id_ville": "int16",
    "ville": "string",
    "type_batiment": "string",
    "n_pieces": "int16",
    "surface_habitable": "float32",
    "surface_terrains_sols": "string"
}

DTYPES_STATELESS_PROCESSED = {
    "n_pieces": "int16",
    "year_month_numeric": "int32",
    "month_sin": "float64",
    "month_cos": "float64",
    "departement": "string",
    "unique_city_id": "string",
    "type_batiment": "string",
    "log_price/m²": "float64",
    "living_area": "float32",
    "surface_terrains_sols": "string"
}

DTYPES_MERGED= {
    "n_tax_households": "float64",
    "average_tax_income": "float64",
    "new_mortgages": "float64",
    "debt_ratio": "float32",
    "interest_rates": "float32",
    "n_rooms": "int16",
    "year_month_numeric": "int32",
    "month_sin": "float64",
    "month_cos": "float64",
    "departement": "string",
    "unique_city_id": "string",
    "building_type": "string",
    "outdoor_area": "string",
    "log_price/m²": "float64",
    "living_area": "float32"
}

DTYPES_PREPROCESSED= {
    "n_tax_households": "float64",
    "average_tax_income": "float64",
    "new_mortgages": "float64",
    "debt_ratio": "float32",
    "interest_rates": "float32",
    "n_rooms": "float32",
    "year_month_numeric": "int32",
    "month_sin": "float64",
    "month_cos": "float64",
    "departement": "category",
    "unique_city_id": "category",
    "no_garden": "category",
    "small_outdoor_space": "category",
    "average_outdoor_space": "category",
    "large_outdoor_space": "category",
    "building_type": "category",
    "log_price/m²": "float64",
    "living_area": "float32",
}

DTYPES_KERAS= {
    "n_tax_households": "float64",
    "average_tax_income": "float64",
    "new_mortgages": "float64",
    "debt_ratio": "float64",
    "interest_rates": "float64",
    "n_rooms": "float32",
    "year_month_numeric": "int32",
    "month_sin": "float64",
    "month_cos": "float64",
    "no_garden": "int8",
    "small_outdoor_space": "int8",
    "average_outdoor_space": "int8",
    "large_outdoor_space": "int8",
    "building_type": "int8",
    "living_area": "float32",
}
