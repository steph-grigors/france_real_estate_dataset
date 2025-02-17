import numpy as np
import pandas as pd
import joblib

import ipdb


from colorama import Fore, Style

from real_estate.ml_logic import *
from real_estate.ml_logic.data_extraction import *
from real_estate.ml_logic.data import *
from real_estate.ml_logic.preprocessor import *
from real_estate.ml_logic.model import xgboost_model, train_xgb_model, initialize_keras_model, compile_keras_model, train_keras_model, evaluate_model
from real_estate.ml_logic.registry import save_model, save_results, load_model

from real_estate.params import *
from real_estate.utils import *



def data_extraction():

    # Extract the transactions by chunk from the .npz file and save them in an output_dir
    process_npz_chunkwise_to_csv(NPZ_FILE_PATH, DATA_EXTRACTION_CHUNK_SIZE, RAW_DATASET_CHUNKS_DIR)

    # Optional step - concatenate the processed chunks into a single .csv file (may take some time)
    concatenate_csv_files(RAW_DATASET_CHUNKS_DIR, RAW_DATASET_OUTPUT_FILE)


def cleaning_in_chunks() -> None:

    raw_dataset_exists = os.path.isfile(RAW_DATASET_OUTPUT_FILE) and os.path.getsize(RAW_DATASET_OUTPUT_FILE) > 0
    cleaned_dataset_exists =  os.path.isfile(CLEANED_DATASET_FILE) and os.path.getsize(CLEANED_DATASET_FILE) > 0

#######################################  CLEANING TRANSACTIONS MAIN DF ####################################################################

    # Assigning local path for saving the cleaned transactions DataFrame
    if cleaned_dataset_exists:

        clean_transactions_df = pd.read_csv(CLEANED_DATASET_FILE, chunksize=CHUNK_SIZE)
        total_rows = sum(1 for _ in open(CLEANED_DATASET_FILE))

        if not (7000000 < total_rows < 9000000):
            raise ValueError(Fore.RED + f"⚠️ Total rows {total_rows} is outside the expected range (7,000,000 to 9,000,000)." + Style.RESET_ALL)

        else:
            print("✅ Skipping cleaning as the cleaned dataset already exists.")

        print(Fore.YELLOW + f'📁 Cleaned transactions DataFrame fetched from cache - Total #rows =  {total_rows}' + Style.RESET_ALL)


    else:
        if raw_dataset_exists:
            print("📁 Raw transactions DataFrame iterable fetched from local CSV...")
            chunks = None
            chunks = pd.read_csv(RAW_DATASET_OUTPUT_FILE, chunksize=CHUNK_SIZE)

            total_rows = sum(1 for _ in open(RAW_DATASET_OUTPUT_FILE))
            total_chunks = total_rows // CHUNK_SIZE

            if os.path.exists(CITY_MAPPING_PATH):
                os.remove(CITY_MAPPING_PATH)

            for chunk_id, chunk in enumerate(chunks):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already cleaned {chunk_id + 1} chunks out of {total_chunks}...")

                # Aplying clean_transactions() (data.py) function and create_city_mapping() (utils.py) function in chunks of CHUNKSIZE
                clean_chunk = clean_transactions(chunk)
                mapping = create_city_mapping(clean_chunk .copy()).reset_index(drop=True)
                clean_processed_chunk = stateless_preprocessor(clean_chunk)

                # Saving to .csv , appending if file exists, writing it file doesn't exist
                if not cleaned_dataset_exists:
                        clean_processed_chunk.to_csv(CLEANED_DATASET_FILE, mode='w', header=True, index=False)
                        cleaned_dataset_exists = True
                else:
                        clean_processed_chunk.to_csv(CLEANED_DATASET_FILE, mode='a', header=False, index=False)

                mapping.to_csv(CITY_MAPPING_PATH,
                   mode='w' if chunk_id == 0 else 'a',
                   header=(chunk_id == 0),
                   index=False)

        else:
            raise Exception(Fore.RED +"⚠️ Please first run data_extraction() function to extract the data from the .npz file." + Style.RESET_ALL)


    clean_city_mapping = pd.read_csv(CITY_MAPPING_PATH)
    clean_city_mapping.drop_duplicates(inplace=True, ignore_index=True)
    clean_city_mapping.to_csv(CITY_MAPPING_PATH, header=True, index=False)

    print(f"✅ Cleaned city mapping saved")
    print(f'✅ Transactions DataFrame cleaned and saved - Total #rows =  {total_rows}')

#######################################  CLEANING SECONDARY DFs ####################################################################

    # Cleaning secondary DataFrames using clean_data() function from data.py
    cleaned_dataframes_dictionnary = clean_data()

    print(f'✅ Secondary DataFrames cleaned and saved')

    # clean_new_mortgages_df = cleaned_dataframes_dictionnary['flux_nouveaux_emprunts_df']
    # clean_tax_households_df = cleaned_dataframes_dictionnary['foyers_fiscaux_df']
    # clean_interest_rates_df = cleaned_dataframes_dictionnary['taux_interet_df']
    # clean_debt_ratio_df = cleaned_dataframes_dictionnary['taux_endettement_df']

    print('🎉 Cleaning and mapping done')
    print('🎉 You can now preprocess the DataFrames before being able to train a model!\n')


def preprocessing_in_chunks() -> None:

    clean_new_mortgages_df = pd.read_csv(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_flux_nouveaux_emprunts_df.csv'))
    clean_tax_households_df = pd.read_csv(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_foyers_fiscaux_df.csv'))
    clean_interest_rates_df = pd.read_csv(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_taux_interet_df.csv'))
    clean_debt_ratio_df = pd.read_csv(os.path.join(CLEANED_DATASET_FOLDER,'cleaned_taux_endettement_df.csv'))

    #######################################  MERGING DATAFRAMES ####################################################################

    cleaned_dataset_exists =  os.path.isfile(CLEANED_DATASET_FILE) and os.path.getsize(CLEANED_DATASET_FILE) > 0
    merged_dataset_exists = os.path.isfile(MERGED_DATASET_FILE) and os.path.getsize(MERGED_DATASET_FILE) > 0

    temporal_features_df = combined_temporal_features_df(clean_new_mortgages_df,
                                                         clean_interest_rates_df,
                                                         clean_debt_ratio_df)


    if merged_dataset_exists:
        merged_transactions_df = pd.read_csv(MERGED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_MERGED,  on_bad_lines='warn')
        total_rows = sum(1 for _ in open(MERGED_DATASET_FILE))
        print(f'📁 Merged transactions DataFrame fetched from cache- Total #rows =  {total_rows}')

    else:
        if cleaned_dataset_exists:
            print("📁 Loading Cleaned DataFrame iterable for processing from local CSV...")
            clean_dataset_df = pd.read_csv(CLEANED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_STATELESS_PROCESSED,  on_bad_lines='warn')
            merged_chunks = []
            total_rows = sum(1 for _ in open(CLEANED_DATASET_FILE)) - 1
            total_chunks = total_rows // CHUNK_SIZE

            for chunk_id, chunk in enumerate(clean_dataset_df):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already merged {chunk_id + 1} chunks out of {total_chunks}...")

                # Merge each chunk with temporal_features_df on the primary key
                merged_chunk = merged_dfs(chunk,
                                        temporal_features_df,
                                        clean_tax_households_df,
                                        primary_keys = ('year_month_numeric', 'unique_city_id'))
                merged_chunks.append(merged_chunk)

            X_merged = pd.concat(merged_chunks, axis=0, ignore_index=True)
            X_merged.columns = ['year_month_numeric','month_sin','month_cos','departement','unique_city_id', 'log_price_per_m2',
                                'living_area', 'building_type', 'n_rooms', 'outdoor_area',
                            'new_mortgages','debt_ratio','interest_rates','n_tax_households','average_tax_income'
                                ]
            X_merged.to_csv(MERGED_DATASET_FILE, index=False)

#######################################  TRAIN-TEST SPLIT BEFORE FINAL PREPROCESSING ####################################################################

            X_merged = X_merged.sort_values(by='year_month_numeric', ascending=True)
            X_merged.dropna(subset=['n_tax_households','average_tax_income'], axis=0, inplace=True)

            # Splits the dataset into training and testing sets with a 70-30 ratio.
            # The split is performed in an ordered manner to preserve the temporal sequence of the data, ensuring that future data points do not appear in the training set.
            train_size  = int(0.7 * len(X_merged))

            train_set = X_merged.iloc[:train_size].copy()
            test_set = X_merged.iloc[train_size:].copy()

            train_set.to_csv(MERGED_TRAIN_FILE, index=False)
            test_set.to_csv(MERGED_TEST_FILE, index=False)

        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the DataFrames have already been cleaned before attempting to further merge them." + Style.RESET_ALL)

        print(f'✅ The DataFrames have successfully been merged! - {X_merged.shape}')
        print(f'✅ Train and Test sets have successfully been saved to .csv!')

#######################################  STATEFUL PREPROCESSOR ####################################################################

# >>>>>>>>>>>>>>>>>>>>TRAIN SET
    train_processed_exists = os.path.isfile(PREPROCESSED_TRAIN_FILE)
    train_set_exists=  os.path.isfile(MERGED_TRAIN_FILE)
    test_set_exists = os.path.isfile(MERGED_TEST_FILE)
    test_processed_exists = os.path.isfile(PREPROCESSED_TEST_FILE)

    if train_processed_exists and test_processed_exists:
        print("✅ Skipping processing as the preprocessed train dataframe already exists.")

    else:
        if train_set_exists:
            print("📁 Loading Train DataFrame iterable for processing from local CSV...")
            chunks = None
            chunks = pd.read_csv(MERGED_TRAIN_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_MERGED, on_bad_lines='warn')

            total_rows = sum(1 for _ in open(MERGED_TRAIN_FILE))
            total_chunks = total_rows // CHUNK_SIZE

            for chunk_id, chunk in enumerate(chunks):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already processed {chunk_id + 1} chunks out of {total_chunks}...")

                preprocessed_train_chunk, fitted_preprocessor = post_merging_preprocessor(chunk, fit=True)
                joblib.dump(fitted_preprocessor, PREPROCESSOR_PATH)

                # Saving to .csv , appending if file exists, writing it file doesn't exist
                if not train_processed_exists:
                    preprocessed_train_chunk.to_csv(PREPROCESSED_TRAIN_FILE, mode='w', header=True, index=False)
                    train_processed_exists = True
                else:
                    preprocessed_train_chunk.to_csv(PREPROCESSED_TRAIN_FILE, mode='a', header=False, index=False)

            total_rows = sum(1 for _ in open(PREPROCESSED_TRAIN_FILE))
            print(f'✅ Train set processed - Total #rows =  {total_rows}')

        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the have correctly been merged before attempting to finalize the preprocessing." + Style.RESET_ALL)

# >>>>>>>>>>>>>>>>>>>>TEST SET
        if test_processed_exists:
            print("✅ Skipping processing as the preprocessed test dataframe already exists.")

        else:
            if test_set_exists:
                print("📁 Loading Test DataFrame iterable for processing from local CSV...")
                chunks = None
                chunks = pd.read_csv(MERGED_TEST_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_MERGED, on_bad_lines='warn')

                total_rows = sum(1 for _ in open(MERGED_TEST_FILE))
                total_chunks = total_rows // CHUNK_SIZE

                for chunk_id, chunk in enumerate(chunks):
                    if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                        print(f"Already processed {chunk_id + 1} chunks out of {total_chunks}...")

                    preprocessed_test_chunk, _ = post_merging_preprocessor(chunk, preprocessor=fitted_preprocessor, fit=False)

                    # Saving to .csv , appending if file exists, writing it file doesn't exist
                    if not test_processed_exists:
                        preprocessed_test_chunk.to_csv(PREPROCESSED_TEST_FILE, mode='w', header=True, index=False)
                        test_processed_exists = True
                    else:
                        preprocessed_test_chunk.to_csv(PREPROCESSED_TEST_FILE, mode='a', header=False, index=False)

                total_rows = sum(1 for _ in open(PREPROCESSED_TEST_FILE))
                print(f'✅ Test Set processed - Total #rows =  {total_rows}')

            else:
                raise Exception(Fore.RED + "⚠️ Please make sure the have correctly been merged before attempting to finalize the preprocessing." + Style.RESET_ALL)

    print(f'📁 Processed train/test sets fetched from cache')
    print("🎉 preprocessing() done")
    print(f'🎉 You can now move forward and train your desired model on the preprocessed dataset!')


def training(model_type= MODEL_TYPE):

    preprocessed_train = pd.read_csv(PREPROCESSED_TRAIN_FILE, dtype=DTYPES_PREPROCESSED)

    y = preprocessed_train.pop('log_price_per_m2')
    X = preprocessed_train


   # Further splits the training dataset into training and validation sets with a 80-20 ratio.
    val_split_index = int(0.8 * len(X))

    X_train, X_val = X.iloc[:val_split_index], X.iloc[val_split_index:]
    y_train, y_val = y.iloc[:val_split_index], y.iloc[val_split_index:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

    if model_type == 'xgboost':

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

        model, metrics = train_xgb_model(params=params,
                                X = X_train, y = y_train,
                                X_val = X_val, y_val = y_val,
                                eval_metric="rmse",
                                early_stopping_rounds=5,
                                verbose=True,
                                )

        val_rmse = np.min(metrics['validation']['rmse'])

        print(f"✅ Model XgBoost trained with a val_rmse of: {val_rmse}")

        training_params = dict(
            context="train",
            row_count=len(X_train),
            params = params
        )

        # Save results on the hard drive using real_estate.ml_logic.registry
        save_results(params=training_params, metrics=dict(rmse=val_rmse), model_type = MODEL_TYPE)

        # Save model weight on the hard drive (and optionally on GCS too)
        save_model(model=model, model_type=MODEL_TYPE)

        return val_rmse

    elif model_type == 'keras':

        model = None

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

        X_departement = X_train_categorical[:, 0]
        X_unique_city = X_train_categorical[:, 1]

        X_departement_val = X_val_categorical[:, 0]
        X_unique_city_val = X_val_categorical[:, 1]

        learning_rate = 0.001
        batch_size = 256
        patience=5

        if model == None:
            model = initialize_keras_model(n_numeric_features=X_train_numeric.shape[1])

        model = compile_keras_model(model, learning_rate)

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

        # Save model and training params
        params = dict(
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
        )

        # Save results & model
        save_results(params=params, metrics=dict(mae=val_rmse), model_type = MODEL_TYPE)
        save_model(model=model, model_type = MODEL_TYPE)

    else:
        raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)


    print("🎉 training() done")
    print(f"🎉 {model_type} model has been successfully trained!")

def evaluate(model_type=MODEL_TYPE):

    preprocessed_test = pd.read_csv(PREPROCESSED_TEST_FILE, dtype=DTYPES_PREPROCESSED)

    y_new = preprocessed_test.pop('log_price_per_m2')
    X_new = preprocessed_test

    model = load_model(model_type)
    assert model is not None

    model = compile_keras_model(model, 0.001)


    X_test_categorical = keras_preprocessor(X_new).values

    categorical_columns = ['building_type',
                                'average_outdoor_space',
                                'large_outdoor_space',
                                'no_garden',
                                'small_outdoor_space',
                                ]

    for col in categorical_columns:
            X_new[col] = X_new[col].cat.codes

    X_test_numeric= X_new.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

    X_departement = X_test_categorical[:, 0]
    X_unique_city = X_test_categorical[:, 1]

    X=[X_departement, X_unique_city, X_test_numeric]
    y=y_new

    metrics_dict = evaluate_model(model=model, X=X, y=y)
    rmse = metrics_dict["rmse"]

    params = dict(
        context="evaluate", # Package behavior
        row_count=len(X_new)
    )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return rmse



def predict(X_pred: pd.DataFrame = None):

    if X_pred is None:
        X_pred = pd.DataFrame(pd.read_csv(MERGED_TEST_FILE, dtype=DTYPES_MERGED).iloc[0, :]).T

    model = load_model(model_type=MODEL_TYPE)
    assert model is not None

    model = compile_keras_model(model, learning_rate=0.001)

    fitted_preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_pred_processed, _ = post_merging_preprocessor(X_pred, preprocessor=fitted_preprocessor, fit=False)
    y_true = X_pred_processed['log_price_per_m2']
    X_pred_processed = X_pred_processed.drop('log_price_per_m2', axis=1)

    X_pred_categorical = keras_preprocessor(X_pred_processed).values

    categorical_columns = ['building_type',
                                'average_outdoor_space',
                                'large_outdoor_space',
                                'no_garden',
                                'small_outdoor_space',
                                ]

    for col in categorical_columns:
            X_pred_processed[col] = X_pred_processed[col].cat.codes

    X_pred_numeric= X_pred_processed.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

    X_pred_departement = X_pred_categorical[:, 0]
    X_pred_unique_city = X_pred_categorical[:, 1]

    X_pred=[X_pred_departement, X_pred_unique_city, X_pred_numeric]
    y_pred = model.predict(X_pred)

    print("\n✅ Prediction done: ", np.exp(y_pred.squeeze()))
    print(f"✅ True value of the property: {np.exp(y_true.squeeze())}")
    print(f"✅ Error: {np.abs(np.exp(y_true.squeeze()) - np.exp(y_pred.squeeze()))}")

    return y_pred


if __name__ == '__main__':
    cleaning_in_chunks()
    preprocessing_in_chunks()
    training(model_type=MODEL_TYPE)
    evaluate(model_type=MODEL_TYPE)
    predict(pd.DataFrame(pd.read_csv(MERGED_TEST_FILE, dtype=DTYPES_MERGED).iloc[75, :]).T)
