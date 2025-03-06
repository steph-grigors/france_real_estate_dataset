import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='/prod/.env')

##################  VARIABLES  #####################
DATA_SIZE = os.environ.get("DATA_SIZE")
DATA_EXTRACTION_CHUNK_SIZE  = int(os.environ.get("DATA_EXTRACTION_CHUNK_SIZE", 50000))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 50000))
MODEL_TARGET = os.environ.get("MODEL_TARGET")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_REGION = os.environ.get("BQ_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
INSTANCE = os.environ.get("INSTANCE")
MODEL_TYPE = os.environ.get("MODEL_TYPE")

##################  CONSTANT PATHS  #####################

# Check if running inside the container
if os.getenv('ENVIRONMENT') == 'container':
    LOCAL_PROJECT_PATH = "/prod"
else:
    LOCAL_PROJECT_PATH = os.path.join(os.path.expanduser('~'), "code", "steph-grigors", "real_estate_dataset")


LOCAL_DATA_PATH = os.path.join(LOCAL_PROJECT_PATH,"data")
LOCAL_REGISTRY_PATH = os.path.join(LOCAL_PROJECT_PATH, "artifacts")

RAW_DATASET_FOLDER = os.path.join(LOCAL_PROJECT_PATH,"data", 'raw_dataset')
CLEANED_DATASET_FOLDER = os.path.join(LOCAL_PROJECT_PATH,"data", 'cleaned_dataset')
MERGED_DATASET_FOLDER = os.path.join(LOCAL_PROJECT_PATH,"data", 'merged_dataset')
PREPROCESSED_DATASET_FOLDER = os.path.join(LOCAL_PROJECT_PATH,"data", 'preprocessed_dataset')



##################  DATA EXTRACTION & PROCESSING PATHS  #####################
NPZ_FILE_PATH = os.path.join(RAW_DATASET_FOLDER, "transactions.npz")
RAW_DATASET_CHUNKS_DIR = os.path.join(RAW_DATASET_FOLDER, "raw_dataset_chunks")
RAW_DATASET_OUTPUT_FILE = os.path.join(RAW_DATASET_FOLDER, "raw_dataset_full", "transactions.csv")

CLEANED_DATASET_FILE = os.path.join(CLEANED_DATASET_FOLDER, "cleaned_transactions_df.csv")
MERGED_DATASET_FILE = os.path.join(MERGED_DATASET_FOLDER, "merged_dataset.csv")

X_TRAIN = os.path.join(MERGED_DATASET_FOLDER,"train_set", "X_train.csv")
Y_TRAIN = os.path.join(MERGED_DATASET_FOLDER,"train_set", "y_train.csv")
X_TEST = os.path.join(MERGED_DATASET_FOLDER, "test_set", "X_test.csv")
Y_TEST = os.path.join(MERGED_DATASET_FOLDER, "test_set", "y_test.csv")

X_TRAIN_PREPROC = os.path.join(PREPROCESSED_DATASET_FOLDER, "X_train_preprocessed.csv")
X_TEST_PREPROC = os.path.join(PREPROCESSED_DATASET_FOLDER, "X_test_preprocessed.csv")


##################  ARTIFACTS & TRAINING PARAMS PATH   #####################
PREPROCESSOR_PATH = os.path.join(LOCAL_REGISTRY_PATH, "preprocessor","preprocessor.pkl")
CITY_MAPPING_PATH = os.path.join(LOCAL_REGISTRY_PATH, "city_mapping.csv")




##################  DTYPES FOR MAPPING  #####################
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
    "log_price_per_m2": "float64",
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
