## __Real Estate in France from 2014 to 2024__

### **Overview**

This project leverages a rich dataset containing nearly all real estate transactions in France from 2014 to today. It includes key economic indicators such as household income levels, average debt and savings, loan interest rates, rental prices, and housing availability per city.
The main challenge is the amount of transactions stored in the main file that approximate 10 Million observations. The objective is to extract, clean, and preprocess the data to prepare it for machine learning model training. The pipeline is designed to work efficiently with large datasets, leveraging Parquet files for performance.

Released under a permissive license, this dataset facilitates research on real estate price dynamics and is available for both academic and commercial applications.

https://www.kaggle.com/datasets/benoitfavier/immobilier-france?resource=download


### **Project Structure**

```
project-root
├── artifacts
│   ├── metrics
│   │   ├── keras
│   │   └── xgboost
│   ├── models
│   │   ├── keras
│   │   └── xgboost
│   ├── params
│   │   ├── keras
│   │   └── xgboost
│   └── preprocessor
├── data
│   ├── cleaned_dataset
│   ├── merged_dataset
│   │   ├── test_set
│   │   └── train_set
│   ├── preprocessed_dataset
│   └── raw_dataset
│       ├── raw_dataset_chunks_csv
│       └── raw_dataset_full
├── Dockerfile
├── Makefile
├── notebooks
├── README.md
├── real_estate_parquet
│   ├── api
│   │   ├── api.py
│   ├── interface
│   │   └── main.py
│   ├── ml_logic
│   │   ├── data_extraction.py
│   │   ├── data.py
│   │   ├── encoders.py
│   │   ├── model.py
│   │   ├── preprocessor.py
│   │   └── registry.py
│   ├── params.py
│   └── utils.py
├── requirements_prod.txt
├── requirements.txt
└── setup.py
```


### **Dependencies**
The following packages are required to run the project:

```
colorama
numpy
pandas
scikit-learn
scipy
statsmodels
tensorflow
xgboost
protobuf
h5py
google-cloud-storage
pyarrow
fastapi
pytz
uvicorn
python-dotenv
```

You can install the dependencies using:
```
pip install -r requirements.txt
```


### **Makefile Overview**

The Makefile provides commands to streamline package management, data processing, model training, and deployment.

```
### Package Management

make reinstall_package – Reinstalls the real-estate package.
make run_work_env – Displays the current environment.
Data Pipeline

make run_extract – Extracts raw data.
make run_clean – Cleans data in chunks.
make run_preproc – Preprocesses data in chunks.
make run_train – Trains the model.
make run_evaluate – Evaluates the model.
make run_predict – Runs predictions.
make run_all – Executes cleaning, preprocessing, training, evaluation, and prediction sequentially.

### Deployment

make run_create_docker_image – Builds the Docker image.
make run_build_docker_container – Runs the API in a Docker container.
make run_api – Starts the FastAPI server locally.

### Setup & Cleanup

make init-directories – Creates necessary project directories.
make clean – Removes temporary and cache files.
```


### **Data Pipeline Workflow**

# Project Pipeline

## 1) Data Extraction

**Input**: .npz file containing raw tabular data.
**Output**: .parquet file (`raw_dataset.parquet`) for efficient storage and processing.

**Steps**:
- Load the .npz file using NumPy.
- Convert the dataset into a Pandas DataFrame.
- Save the DataFrame in Parquet format for optimized processing.

---

## 2) Data Cleaning (Chunk-Based Processing)

**Input**: `raw_dataset.parquet`.
**Output**: `cleaned_dataset.parquet`.

**Steps**:
- Load the raw dataset in chunks to handle large file sizes efficiently.
- Perform type casting according to predefined `DTYPES_RAW`.
- Apply city mappings to standardize city names.
- Remove duplicates and handle missing values.
- Save the cleaned data back to Parquet format in chunks, merging them at the end.

---

## 3) Data Preprocessing (Feature Engineering & Splitting)

**Input**: `cleaned_dataset.parquet`.
**Output**: `preprocessed_train.parquet`, `preprocessed_test.parquet`.

**Steps**:
- Load the cleaned dataset in chunks.
- Merge with external datasets for additional feature enrichment.
- Apply train-test split while ensuring stratification.
- Perform feature transformations, including:
  - One-hot encoding for categorical features.
  - Scaling numerical features using MinMax or Standard Scaler.
- Save the final train and test sets in Parquet format.

---

## 4) Model Training

**Input**: `preprocessed_train.parquet`.
**Output**: Trained model saved locally or in cloud storage.

**Steps**:
- Load preprocessed training data.
- Split into training and validation sets (80-20 split).
- Train one of the following models:
  - XGBoost: Uses decision trees for regression.
  - Keras Neural Network: Incorporates embeddings for categorical variables.
- Save model weights and training results for reproducibility.

---

## 5) Model Evaluation

**Input**: `preprocessed_test.parquet`.
**Output**: Model evaluation metrics (RMSE, MAE, etc.).

**Steps**:
- Load the test dataset.
- Preprocess categorical and numerical features accordingly.
- Load the trained model and evaluate it on the test set.
- Log and save evaluation metrics for further analysis.

---

## 6) Model Prediction

**Input**: New data (`X_pred`).
**Output**: Predicted real estate price per square meter.

**Steps**:
- Load the trained model.
- Preprocess the input data using the same pipeline as training.
- Generate predictions and return the results.
