import numpy as np
import pandas as pd
import joblib

import ipdb


from colorama import Fore, Style

from real_estate_parquet.ml_logic import *
from real_estate_parquet.ml_logic.data_extraction import *
from real_estate_parquet.ml_logic.data import *
from real_estate_parquet.ml_logic.preprocessor import *
from real_estate_parquet.ml_logic.model import xgboost_model, train_xgb_model, initialize_keras_model, compile_keras_model, train_keras_model, evaluate_model
from real_estate_parquet.ml_logic.registry import save_model, save_results, load_model

from real_estate_parquet.params import *
from real_estate_parquet.utils import *



def data_extraction():
    """
    Extracts transactions data from an .npz file and saves it as a Parquet file.
    """

    # File paths and configuration
    npz_file_path = os.path.join(LOCAL_DATA_PATH, "raw_dataset", "transactions.npz")
    output_file = os.path.join(LOCAL_DATA_PATH, "raw_dataset_full", "transactions.parquet")

    # Extract the transactions by chunk from the .npz file and save them in an output_file
    process_npz_to_parquet(npz_file_path, output_file, chunk_size= CHUNK_SIZE)


def cleaning_in_chunks() -> None:
    """
    Cleans transaction data in chunks and saves the cleaned dataset.
    Ensures that the data is properly transformed and stored in a Parquet file.
    The function processes data in batches to manage memory usage and efficiency.
    """

    # Check if raw and cleaned datasets already exist in the specified paths
    raw_dataset_exists = os.path.isfile(RAW_DATASET_OUTPUT_FILE) and os.path.getsize(RAW_DATASET_OUTPUT_FILE) > 0
    cleaned_dataset_exists = os.path.isfile(CLEANED_DATASET_FILE) and os.path.getsize(CLEANED_DATASET_FILE) > 0

#######################################  CLEANING TRANSACTIONS MAIN DF ####################################################################

    # If the cleaned dataset exists, load it from cache and validate its row count
    if cleaned_dataset_exists:
        clean_transactions_df = pd.read_parquet(CLEANED_DATASET_FILE)
        total_rows = len(clean_transactions_df)

        # Ensure that the number of rows falls within the expected range (7M to 9M)
        if not (7000000 < total_rows < 9000000):
            raise ValueError(Fore.RED + f"⚠️ Total rows {total_rows} is outside the expected range (7,000,000 to 9,000,000)." + Style.RESET_ALL)
        else:
            print("✅ Skipping cleaning as the cleaned dataset already exists.")

        print(Fore.YELLOW + f'📁 Cleaned transactions DataFrame fetched from cache - Total #rows =  {total_rows}' + Style.RESET_ALL)

    else:
        # If the cleaned dataset does not exist, check if the raw dataset is available
        if raw_dataset_exists:
            print("📁 Raw transactions DataFrame iterable fetched from local Parquet...")

            cleaned_batches = []  # List to store cleaned data batches
            city_mappings = []  # List to store city mapping batches

            # If any existing city mapping or cleaned dataset files exist, delete them before starting
            if os.path.exists(CITY_MAPPING_PATH):
                os.remove(CITY_MAPPING_PATH)
            if os.path.exists(CLEANED_DATASET_FILE):
                os.remove(CLEANED_DATASET_FILE)

            # Clean and save each batch of raw data
            for batch_id, batch in enumerate(pq.ParquetFile(RAW_DATASET_OUTPUT_FILE).iter_batches(batch_size=CHUNK_SIZE)):
                total_rows = pq.ParquetFile(RAW_DATASET_OUTPUT_FILE).metadata.num_rows
                total_batches = total_rows // CHUNK_SIZE
                raw_batch = batch.to_pandas()
                raw_batch = raw_batch.astype(DTYPES_RAW)  # Cast to raw data types for efficient processing

                # Print progress every 10 batches or at the final batch
                if (batch_id + 1) % 10 == 0 or batch_id == total_batches:
                    print(f"Already processed {batch_id + 1} chunks out of {total_batches}...")

                # Clean the raw batch
                clean_batch = clean_transactions(raw_batch)

                # Generate city mappings for the current batch and add them to the list
                city_mapping = create_city_mapping(clean_batch)
                city_mappings.append(city_mapping)

                # Apply stateless preprocessing (e.g., scaling) to the cleaned batch
                clean_processed_batch = stateless_preprocessor(clean_batch)
                cleaned_batches.append(clean_processed_batch)

            # Concatenate and save all cleaned data batches into a single Parquet file
            cleaned_df = pd.concat(cleaned_batches, ignore_index=True)
            cleaned_df.to_parquet(CLEANED_DATASET_FILE, engine="pyarrow", index=False)

            # Concatenate and save city mappings, ensuring no duplicates, into a separate Parquet file
            city_mapping_df = pd.concat(city_mappings, ignore_index=True).drop_duplicates()
            city_mapping_df.to_parquet(CITY_MAPPING_PATH, engine="pyarrow", index=False)

        else:
            # Raise an exception if the raw dataset is not available
            raise Exception(Fore.RED +"⚠️ Please first run data_extraction() function to extract the data from the .npz file." + Style.RESET_ALL)

    # Apply final cleaning steps to the city mapping and save to Parquet (removing duplicates)
    clean_city_mapping = pd.read_parquet(CITY_MAPPING_PATH)
    clean_city_mapping.drop_duplicates(inplace=True, ignore_index=True)
    clean_city_mapping.to_parquet(CITY_MAPPING_PATH, index=False)

    # Print success messages
    print(f"✅ Cleaned city mapping saved")
    print(f'✅ Transactions DataFrame cleaned and saved - Total #rows =  {total_rows}')


#######################################  CLEANING SECONDARY DFs ####################################################################

    # Clean secondary DataFrames (non-primary) using the clean_data function from data.py
    clean_data()

    # Print final success message
    print(f'✅ Secondary DataFrames cleaned and saved')
    print('🎉 Cleaning and mapping done')
    print('🎉 You can now preprocess the DataFrames before being able to train a model!\n')


def preprocessing_in_chunks() -> None:
    """
    Preprocesses cleaned data in chunks and merges it with other datasets.
    Ensures the dataset is ready for model training by saving train-test splits.
    """

    # Load cleaned datasets from Parquet files into DataFrames
    clean_new_mortgages_df = pd.read_parquet(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_flux_nouveaux_emprunts_df.parquet'))
    clean_tax_households_df = pd.read_parquet(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_foyers_fiscaux_df.parquet'))
    clean_interest_rates_df = pd.read_parquet(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_taux_interet_df.parquet'))
    clean_debt_ratio_df = pd.read_parquet(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_taux_endettement_df.parquet'))

    #######################################  MERGING DATAFRAMES ####################################################################

    # Check if cleaned and merged datasets already exist on disk
    cleaned_dataset_exists =  os.path.isfile(CLEANED_DATASET_FILE) and os.path.getsize(CLEANED_DATASET_FILE) > 0
    merged_dataset_exists = os.path.isfile(MERGED_DATASET_FILE) and os.path.getsize(MERGED_DATASET_FILE) > 0

    # Combine temporal features from the datasets
    temporal_features_df = combined_temporal_features_df(clean_new_mortgages_df,
                                                         clean_interest_rates_df,
                                                         clean_debt_ratio_df)

    # If merged dataset exists, load it from disk
    if merged_dataset_exists:
        merged_transactions_df = pd.read_parquet(MERGED_DATASET_FILE)
        total_rows = len(merged_transactions_df)
        print(f'📁 Merged transactions DataFrame fetched from cache- Total #rows =  {total_rows}')

    else:
        # If cleaned dataset exists, merge the chunks and process them
        if cleaned_dataset_exists:
            print("📁 Loading Cleaned DataFrame iterable for processing from local Parquet file...")

            # If merged dataset exists, remove the previous file
            if os.path.exists(MERGED_DATASET_FILE):
                os.remove(MERGED_DATASET_FILE)

            merged_batches = []  # List to store merged DataFrame batches

            # Process and save each batch
            for batch_id, batch in enumerate(pq.ParquetFile(CLEANED_DATASET_FILE).iter_batches(batch_size=CHUNK_SIZE)):
                total_rows = pq.ParquetFile(CLEANED_DATASET_FILE).metadata.num_rows
                total_batches = total_rows // CHUNK_SIZE
                cleaned_batch = batch.to_pandas()
                cleaned_batch = cleaned_batch.astype(DTYPES_STATELESS_PROCESSED)

                # Print progress every 10 batches or at the last batch
                if (batch_id + 1) % 10 == 0 or batch_id == total_batches:
                    print(f"Already merged {batch_id + 1} chunks out of {total_batches}...")

                # Merge each batch with temporal features DataFrame
                merged_batch = merged_dfs(cleaned_batch,
                                          temporal_features_df,
                                          clean_tax_households_df,
                                          primary_keys=('year_month_numeric', 'unique_city_id'))
                merged_batches.append(merged_batch)

            # Concatenate merged batches into a single DataFrame
            X_merged = pd.concat(merged_batches, axis=0, ignore_index=True)
            # Rename columns to match the required features
            X_merged.columns = ['year_month_numeric','month_sin','month_cos','departement','unique_city_id', 'log_price_per_m2',
                                'living_area', 'building_type', 'n_rooms', 'outdoor_area',
                            'new_mortgages','debt_ratio','interest_rates','n_tax_households','average_tax_income']

            # Save the merged DataFrame to a Parquet file
            X_merged.to_parquet(MERGED_DATASET_FILE, engine='pyarrow', compression='snappy', index=False)

            #######################################  TRAIN-TEST SPLIT BEFORE FINAL PREPROCESSING ####################################################################

            # Sort the DataFrame by the temporal column to preserve the time sequence
            X_merged = X_merged.sort_values(by='year_month_numeric', ascending=True)
            X_merged.dropna(subset=['n_tax_households','average_tax_income'], axis=0, inplace=True)

            # Perform train-test split (70-30) while maintaining the time sequence
            train_size  = int(0.7 * len(X_merged))
            train_set = X_merged.iloc[:train_size].copy()
            test_set = X_merged.iloc[train_size:].copy()

            # Separate features and target variable for train and test sets
            y_train = pd.DataFrame(train_set.pop("log_price_per_m2"))
            X_train = train_set
            y_test = pd.DataFrame(test_set.pop("log_price_per_m2"))
            X_test = test_set

            # Save train and test sets to Parquet files
            for dataset, path in zip([X_train, y_train, X_test, y_test], [X_TRAIN, Y_TRAIN, X_TEST, Y_TEST]):
                dataset.to_parquet(path, engine='pyarrow', compression='snappy', index=False)

        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the DataFrames have already been cleaned before attempting to further merge them." + Style.RESET_ALL)

        print(f'✅ The DataFrames have successfully been merged! - {X_merged.shape}')
        print(f'✅ Train and Test sets have successfully been saved to .Parquet file!')

#######################################  STATEFUL PREPROCESSOR ####################################################################

# >>>>>>>>>>>>>>>>>>>>TRAIN SET
    # Check if preprocessed train set already exists
    X_train_preprocessed_exists = os.path.isfile(X_TRAIN_PREPROC)
    X_train_set_exists = os.path.isfile(X_TRAIN)
    X_test_set_exists = os.path.isfile(X_TEST)
    X_test_preprocessed_exists = os.path.isfile(X_TEST_PREPROC)

    # If preprocessed train set exists, skip the preprocessing
    if X_train_preprocessed_exists and X_test_preprocessed_exists:
        print("✅ Skipping processing as the preprocessed train dataframe already exists.")

    else:
        if X_train_set_exists:
            # Remove previous preprocessed files if they exist
            if os.path.exists(X_TRAIN_PREPROC):
                os.remove(X_TRAIN_PREPROC)

            if os.path.exists(PREPROCESSOR_PATH):
                os.remove(PREPROCESSOR_PATH)

            print("📁 Loading Train DataFrame iterable for processing from local Parquet file...")

            preprocessed_batches = []  # List to store preprocessed DataFrame batches

            # Process and preprocess each batch of the training data
            for batch_id, batch in enumerate(pq.ParquetFile(X_TRAIN).iter_batches(batch_size=CHUNK_SIZE)):
                total_rows = pq.ParquetFile(X_TRAIN).metadata.num_rows
                total_batches = total_rows // CHUNK_SIZE
                X_train_batch = batch.to_pandas()
                X_train_batch = X_train_batch.astype(DTYPES_MERGED)

                # Print progress every 10 batches or at the last batch
                if (batch_id + 1) % 10 == 0 or batch_id == total_batches:
                    print(f"Already preprocessed {batch_id + 1} batches out of {total_batches}...")

                # Apply preprocessing to each batch and fit the preprocessor
                preprocessed_train_batch, fitted_preprocessor = post_merging_preprocessor(X_train_batch, fit=True)
                joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)

                preprocessed_batches.append(preprocessed_train_batch)

            # Concatenate preprocessed batches into a single DataFrame and save it
            preprocessed_df = pd.concat(preprocessed_batches, ignore_index=True)
            preprocessed_df.to_parquet(X_TRAIN_PREPROC, engine='pyarrow', compression='snappy', index=False)

            total_rows = len(preprocessed_df)
            print(f'✅ Train set processed - Total #rows =  {total_rows}')

        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the DataFrames have correctly been merged before attempting to finalize the preprocessing." + Style.RESET_ALL)

# >>>>>>>>>>>>>>>>>>>>TEST SET
    # If preprocessed test set already exists, skip preprocessing
    if X_test_preprocessed_exists:
            print("✅ Skipping processing as the preprocessed test dataframe already exists.")

    else:
        if X_test_set_exists:
            # Remove previous preprocessed test file if it exists
            if os.path.exists(X_TEST_PREPROC):
                os.remove(X_TEST_PREPROC)

            print("📁 Loading Test DataFrame iterable for processing from local Parquet file...")

            preprocessed_batches = []  # List to store preprocessed DataFrame batches

            # Process and preprocess each batch of the test data
            for batch_id, batch in enumerate(pq.ParquetFile(X_TEST).iter_batches(batch_size=CHUNK_SIZE)):
                total_rows = pq.ParquetFile(X_TEST).metadata.num_rows
                total_batches = total_rows // CHUNK_SIZE
                X_test_batch = batch.to_pandas()
                X_test_batch = X_test_batch.astype(DTYPES_MERGED)

                # Print progress every 10 batches or at the last batch
                if (batch_id + 1) % 10 == 0 or batch_id == total_batches:
                    print(f"Already preprocessed {batch_id + 1} batches out of {total_batches}...")

                # Apply preprocessing to each batch using the fitted preprocessor
                preprocessed_test_batch, _ = post_merging_preprocessor(X_test_batch, preprocessor=fitted_preprocessor, fit=False)

                preprocessed_batches.append(preprocessed_test_batch)

            # Concatenate preprocessed test batches into a single DataFrame and save it
            preprocessed_df = pd.concat(preprocessed_batches, ignore_index=True)
            preprocessed_df.to_parquet(X_TEST_PREPROC, engine='pyarrow', compression='snappy', index=False)

            total_rows = len(preprocessed_df)
            print(f'✅ Test set processed - Total #rows =  {total_rows}')

        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the test data has correctly been merged before attempting to finalize the preprocessing." + Style.RESET_ALL)

    print(f'📁 Processed train/test sets fetched from cache')
    print("🎉 preprocessing() done")
    print(f'🎉 You can now move forward and train your desired model on the preprocessed dataset!')


def train_model(model_type= MODEL_TYPE):

    # Load preprocessed training data
    X_train_preproc = pd.read_parquet(X_TRAIN_PREPROC).astype(DTYPES_PREPROCESSED)
    y_train_preproc = pd.read_parquet(Y_TRAIN).astype("float64")

    # Merge target variable into training set temporarily for splitting
    X_train_preproc["log_price_per_m2"] = y_train_preproc
    y = X_train_preproc.pop('log_price_per_m2')
    X = X_train_preproc

   # Further splits the training dataset into training and validation sets with a 80-20 ratio.
    val_split_index = int(0.8 * len(X))
    X_train, X_val = X.iloc[:val_split_index], X.iloc[val_split_index:]
    y_train, y_val = y.iloc[:val_split_index], y.iloc[val_split_index:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    if model_type == 'xgboost':
        """
        Train an XGBoost model with predefined hyperparameters.
        """
        model = None
        params = xgboost_model({
                                # "n_estimators": 50,
                                "learning_rate": 0.05,
                                "max_depth": 8,
                                "subsample": 0.8,
                                "tree_method": "hist",
                                "colsample_bytree": 0.8,
                                "objective": "reg:squarederror",
                                "random_state": 42,
                                "verbosity": 1,
                                'gamma': 0.1,
                                })

        print("✅ Model instantiated")

        # Train the model
        model, metrics = train_xgb_model(params=params,
                                X = X_train, y = y_train,
                                X_val = X_val, y_val = y_val,
                                eval_metric="rmse",
                                early_stopping_rounds=5,
                                verbose=True,
                                )

        val_rmse = np.min(metrics['validation']['rmse'])

        print(f"✅ Model XgBoost trained with a val_rmse of: {val_rmse}")

        # Save training results in a dict
        training_params = dict(
            context="train",
            row_count=len(X_train),
            params = params
        )

        # Save results either locally or on GCS using registry.py
        save_results(params=training_params, metrics=dict(rmse=val_rmse), model_type = MODEL_TYPE)

        # Save model weights either locally or on GCS using registry.py
        save_model(model=model, model_type=MODEL_TYPE)

        return val_rmse

    elif model_type == 'keras':
        """
        Train a Keras model with embeddings for categorical features.
        """
        model = None

        # Process categorical and numerical features separately
        X_train_categorical = keras_preprocessor(X_train).values
        X_val_categorical = keras_preprocessor(X_val).values

        categorical_columns = ['building_type',
                                'average_outdoor_space',
                                'large_outdoor_space',
                                'no_garden',
                                'small_outdoor_space',
                                ]

        for col in categorical_columns:
            X_train[col] = X_train[col].cat.codes
            X_val[col] = X_val[col].cat.codes

        X_train_numeric= X_train.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values
        X_val_numeric= X_val.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

        # Extract categorical embeddings
        X_departement = X_train_categorical[:, 0]
        X_unique_city = X_train_categorical[:, 1]

        X_departement_val = X_val_categorical[:, 0]
        X_unique_city_val = X_val_categorical[:, 1]

        # Set model hyperparameters
        learning_rate = 0.001
        batch_size = 256
        patience=2

        # Initialize and compile the model
        if model == None:
            model = initialize_keras_model(n_numeric_features=X_train_numeric.shape[1])

        model = compile_keras_model(model, learning_rate)

        # Train the model
        model, history = train_keras_model(
            model,
            X={"departement_input": X_departement,
               "unique_city_id_input": X_unique_city,
               "numeric_input": X_train_numeric},
            y=y_train,
            batch_size=batch_size,
            patience=patience,
            validation_data=(
                            {"departement_input": X_departement_val,
                             "unique_city_id_input": X_unique_city_val,
                             "numeric_input": X_val_numeric},
                            y_val
                            )
        )

        val_rmse = np.min(history.history['rmse'])

        print(f"✅ Neural Network trained with a val_rmse of: {round(val_rmse, 2)}")

        # Save training results in a dict
        params = dict(
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
        )

        # Save results either locally or on GCS using registry.py
        save_results(params=params, metrics=dict(mae=val_rmse), model_type = MODEL_TYPE)

        # Save model weights either locally or on GCS using registry.py
        save_model(model=model, model_type = MODEL_TYPE)

    else:
        raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)


    print("🎉 training() done")
    print(f"🎉 {model_type} model has been successfully trained!")

def evaluate_model(model_type=MODEL_TYPE):
    """
    Evaluates a trained model on the test dataset.

    Args:
        model_type (str): Type of model to evaluate ('keras' or another type).

    Returns:
        float: RMSE (Root Mean Squared Error) of the model's performance on the test set.
    """

    # Load preprocessed test data
    X_test_preproc = pd.read_parquet(X_TEST_PREPROC).astype(DTYPES_PREPROCESSED)
    y_test_preproc = pd.read_parquet(Y_TEST).astype("float64")

    # Append target variable to features for proper processing
    X_test_preproc["log_price_per_m²"] = y_test_preproc

    # Separate features and target variable
    y_new = X_test_preproc.pop('log_price_per_m²')
    X_new = X_test_preproc

    # Load the trained model
    model = load_model(model_type)
    assert model is not None

    # Compile the model with the appropriate learning rate
    model = compile_keras_model(model, 0.001)

    # Preprocess categorical features
    X_test_categorical = keras_preprocessor(X_new).values

    categorical_columns = ['building_type',
                                'average_outdoor_space',
                                'large_outdoor_space',
                                'no_garden',
                                'small_outdoor_space',
                                ]

    # Convert categorical columns to category codes for numerical processing
    for col in categorical_columns:
            X_new[col] = X_new[col].cat.codes

    # Separate numeric features from categorical features
    X_test_numeric= X_new.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

    # Extract categorical feature arrays
    X_departement = X_test_categorical[:, 0]
    X_unique_city = X_test_categorical[:, 1]

    # Prepare input data structure
    X=[X_departement, X_unique_city, X_test_numeric]
    y=y_new

    # Evaluate the model and retrieve metrics
    metrics_dict = evaluate_model(model=model, X=X, y=y)
    rmse = metrics_dict["rmse"]

    # Store evaluation results in a dict
    params = dict(
        context="evaluate", # Package behavior
        row_count=len(X_new)
    )

    # Save results either locally or on GCS using registry.py
    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return rmse



def predict(X_pred: pd.DataFrame = None):
    """
    Generates a price per square meter prediction using a trained model.

    Parameters:
    X_pred (pd.DataFrame, optional): Input data for prediction. If None, a sample from X_TEST is used.

    Returns:
    np.array: Predicted price per square meter (log-transformed and exponentiated back).
    """

    # Use a random sample from X_TEST if no input is provided
    if X_pred is None:
        random_X_pred = np.random.randint(low =0, high = len(pd.read_parquet(X_TEST)))
        X_pred = pd.DataFrame(pd.read_parquet(X_TEST).astype(DTYPES_MERGED).iloc[random_X_pred, :]).T

    # Load the trained model
    model = load_model(model_type=MODEL_TYPE)
    assert model is not None

    # Compile the model with a set learning rate
    model = compile_keras_model(model, learning_rate=0.001)

    # Load the fitted preprocessor and preprocess the input data
    fitted_preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_pred_processed, _ = post_merging_preprocessor(X_pred, preprocessor=fitted_preprocessor, fit=False)

    # Convert categorical features using the keras preprocessor
    X_pred_categorical = keras_preprocessor(X_pred_processed).values

    # List of categorical columns that require encoding
    categorical_columns = ['building_type',
                                'average_outdoor_space',
                                'large_outdoor_space',
                                'no_garden',
                                'small_outdoor_space',
                                ]

    # Convert categorical columns to numerical codes
    for col in categorical_columns:
            X_pred_processed[col] = X_pred_processed[col].cat.codes

    # Extract numerical features and convert to required dtype
    X_pred_numeric= X_pred_processed.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

    # Extract categorical embeddings
    X_pred_departement = X_pred_categorical[:, 0]
    X_pred_unique_city = X_pred_categorical[:, 1]

    # Combine preprocessed input features
    X_pred=[X_pred_departement, X_pred_unique_city, X_pred_numeric]

    # Generate predictions
    y_pred = model.predict(X_pred)

    print("\n✅ Prediction done: ", np.exp(y_pred.squeeze()))

    return y_pred


if __name__ == '__main__':
    # cleaning_in_chunks()
    # preprocessing_in_chunks()
    # training(model_type=MODEL_TYPE)
    # evaluate(model_type=MODEL_TYPE)
    predict()
