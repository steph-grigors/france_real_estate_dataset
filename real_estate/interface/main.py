import numpy as np
import pandas as pd
import ipdb

from colorama import Fore, Style

from real_estate.ml_logic import *
from real_estate.ml_logic.data_extraction import *
from real_estate.ml_logic.data import *
from real_estate.ml_logic.preprocessor import *
from real_estate.ml_logic.model import xgboost_model, train_xgb_model, initialize_keras_model, compile_keras_model, train_keras_model
from real_estate.ml_logic.registry import save_model, save_results

from real_estate.params import *
from real_estate.utils import *



def data_extraction():

    # Extract the transactions by chunk from the .npz file and save them in an output_dir
    process_npz_chunkwise_to_csv(NPZ_FILE_PATH, DATA_EXTRACTION_CHUNK_SIZE, RAW_DATASET_CHUNKS_DIR)

    # Optional step - concatenate the processed chunks into a single .csv file (may take some time)
    concatenate_csv_files(RAW_DATASET_CHUNKS_DIR, RAW_DATASET_OUTPUT_FILE)


def cleaning_in_chunks():

    raw_dataset_exists = os.path.isfile(RAW_DATASET_OUTPUT_FILE)
    cleaned_dataset_exists =  os.path.isfile(CLEANED_DATASET_FILE)
    city_mapping_exists = os.path.isfile(CITY_MAPPING_PATH)

#######################################  CLEANING TRANSACTIONS MAIN DF ####################################################################

    # Assigning local path for saving the cleaned transactions DataFrame
    if cleaned_dataset_exists:

        clean_transactions_df = pd.read_csv(CLEANED_DATASET_FILE, chunksize=CHUNK_SIZE)
        total_rows = sum(1 for _ in open(CLEANED_DATASET_FILE))

        if not (7000000 < total_rows < 9000000):
            raise ValueError(Fore.RED + f"⚠️ Total rows {total_rows} is outside the expected range (7,000,000 to 9,000,000)." + Style.RESET_ALL)


        # Cleaning secondary DataFrames using clean_data() function from data.py
        cleaned_dataframes_dictionnary = clean_data()
        clean_new_mortgages_df = cleaned_dataframes_dictionnary['flux_nouveaux_emprunts_df']
        clean_tax_households_df = cleaned_dataframes_dictionnary['foyers_fiscaux_df']
        clean_interest_rates_df = cleaned_dataframes_dictionnary['taux_interet_df']
        clean_debt_ratio_df = cleaned_dataframes_dictionnary['taux_endettement_df']

        print(Fore.YELLOW + f'📁 Cleaned transactions DataFrame fetched from cache - Total #rows =  {total_rows}' + Style.RESET_ALL)


        return clean_transactions_df, clean_new_mortgages_df, clean_tax_households_df, clean_interest_rates_df, clean_debt_ratio_df

    else:
        if raw_dataset_exists:
            print("📁 Raw transactions DataFrame iterable fetched from local CSV...")
            chunks = None
            chunks = pd.read_csv(RAW_DATASET_OUTPUT_FILE, chunksize=CHUNK_SIZE)

            total_rows = sum(1 for _ in open(RAW_DATASET_OUTPUT_FILE))
            total_chunks = total_rows // CHUNK_SIZE

            for chunk_id, chunk in enumerate(chunks):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already cleaned {chunk_id + 1} chunks out of {total_chunks}...")

                # Aplying clean_transactions() (data.py) function and create_city_mapping() (utils.py) function in chunks of CHUNKSIZE
                clean_chunk = clean_transactions(chunk)
                mapping = create_city_mapping(chunk)

                # Saving to .csv , appending if file exists, writing it file doesn't exist
                if not cleaned_dataset_exists:
                        clean_chunk.to_csv(CLEANED_DATASET_FILE, mode='w', header=True, index=False)
                        cleaned_dataset_exists = True
                else:
                        clean_chunk.to_csv(CLEANED_DATASET_FILE, mode='a', header=False, index=False)

                if not city_mapping_exists:
                        mapping.to_csv('city_mapping.csv', mode='w', header=True, index=False)
                        city_mapping_exists = True
                else:
                        mapping.to_csv('city_mapping.csv', mode='a', header=False, index=False)

        else:
            raise Exception(Fore.RED +"⚠️ Please first run data_extraction() function to extract the data from the .npz file." + Style.RESET_ALL)


    # Matching the name of the return variable with the names of the secondary DataFrames
    clean_transactions_df = pd.read_csv(CLEANED_DATASET_FILE, chunksize=CHUNK_SIZE)
    total_rows = sum(1 for _ in open(CLEANED_DATASET_FILE))

    print(f'✅ Transactions DataFrame cleaned - Total #rows =  {total_rows}')

#######################################  CLEANING SECODNARY DFs ####################################################################

    # Cleaning secondary DataFrames using clean_data() function from data.py
    cleaned_dataframes_dictionnary = clean_data()

    clean_new_mortgages_df = cleaned_dataframes_dictionnary['flux_nouveaux_emprunts_df']
    clean_tax_households_df = cleaned_dataframes_dictionnary['foyers_fiscaux_df']
    clean_interest_rates_df = cleaned_dataframes_dictionnary['taux_interet_df']
    clean_debt_ratio_df = cleaned_dataframes_dictionnary['taux_endettement_df']

    print('🎉 Cleaning and mapping done')
    print('🎉 You can now preprocess the DataFrames before being able to train a model!')

    return clean_transactions_df, clean_new_mortgages_df, clean_tax_households_df, clean_interest_rates_df, clean_debt_ratio_df


def preprocessing_in_chunks(clean_transactions_df, clean_new_mortgages_df, clean_tax_households_df, clean_interest_rates_df, clean_debt_ratio_df):

    cleaned_dataset_exists =  os.path.isfile(CLEANED_DATASET_FILE)
    processed_dataset_exists = os.path.isfile(PROCESSED_DATASET_FILE)

#######################################  FIRST PREPROCESSOR ####################################################################

    if processed_dataset_exists:
        processed_transactions_df = pd.read_csv(PROCESSED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_PROCESSED,  on_bad_lines='warn')
        total_rows = sum(1 for _ in open(PROCESSED_DATASET_FILE))

        print(f'📁 Processed transactions DataFrame fetched from cache- Total #rows =  {total_rows}')

    else:
        if cleaned_dataset_exists:
            print("📁 Loading Cleaned DataFrame iterable for processing from local CSV...")
            chunks = None
            chunks = pd.read_csv(CLEANED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_CLEANED, parse_dates=["date_transaction"], on_bad_lines='warn')

            total_rows = sum(1 for _ in open(CLEANED_DATASET_FILE))
            total_chunks = total_rows // CHUNK_SIZE

            for chunk_id, chunk in enumerate(chunks):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already processed {chunk_id + 1} chunks out of {total_chunks}...")

                processed_chunk = pre_merging_preprocessor(chunk)

                # Saving to .csv , appending if file exists, writing it file doesn't exist
                if not processed_dataset_exists:
                    processed_chunk.to_csv(PROCESSED_DATASET_FILE, mode='w', header=True, index=False)
                    processed_dataset_exists = True
                else:
                    processed_chunk.to_csv(PROCESSED_DATASET_FILE, mode='a', header=False, index=False)

        else:
            raise Exception(Fore.RED + "⚠️ Please first run cleaning_in_chunks() function to return a clean DataFrame of transactions." + Style.RESET_ALL)


    # Matching the name of the return variable with the names of the secondary DataFrames
    processed_transactions_df = pd.read_csv(PROCESSED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_PROCESSED,  on_bad_lines='warn')
    total_rows = sum(1 for _ in open(PROCESSED_DATASET_FILE))
    total_chunks = total_rows // CHUNK_SIZE


    print(f'✅ Transactions DataFrame processed - Total #rows =  {total_rows}')

#######################################  MERGING DATAFRAMES ####################################################################


    merged_dataset_exists = os.path.isfile(MERGED_DATASET_FILE)

    temporal_features_df = combined_temporal_features_df(clean_new_mortgages_df,
                                                         clean_interest_rates_df,
                                                         clean_debt_ratio_df)

    if merged_dataset_exists:
        merged_transactions_df = pd.read_csv(MERGED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_PREPROCESSED,  on_bad_lines='warn')
        total_rows = sum(1 for _ in open(MERGED_DATASET_FILE))
        print(f'📁 Merged transactions DataFrame fetched from cache- Total #rows =  {total_rows}')

    else:
        if processed_dataset_exists:
            print("📁 Loading Processed DataFrame iterable for processing from local CSV...")
            merged_chunks = []
            total_rows = sum(1 for _ in open(PROCESSED_DATASET_FILE)) - 1
            total_chunks = total_rows // CHUNK_SIZE

            for chunk_id, chunk in enumerate(processed_transactions_df):
                if (chunk_id + 1) % 10 == 0 or chunk_id == total_chunks:
                    print(f"Already merged {chunk_id + 1} chunks out of {total_chunks}...")

                # Merge each chunk with temporal_features_df on the primary key
                merged_chunk = merged_dfs(chunk,
                                        temporal_features_df,
                                        clean_tax_households_df,
                                        primary_keys = ('year_month_numeric', 'unique_city_id'))

                merged_chunks.append(merged_chunk)

            X_merged = pd.concat(merged_chunks, axis=0, ignore_index=True)
            X_merged.columns = ['n_rooms','year_month_numeric','month_sin','month_cos','departement','unique_city_id',
                            'no_garden','small_outdoor_space','average_outdoor_space','large_outdoor_space','building_type',
                            'price/m²','living_area','new_mortgages','debt_ratio','interest_rates','n_tax_households','average_tax_income'
                            ]

            X_merged.to_csv(MERGED_DATASET_FILE, index=False)
        else:
            raise Exception(Fore.RED + "⚠️ Please make sure the main DataFrame has already been processed before attempting to further merge it." + Style.RESET_ALL)

        print(f'✅ Final DataFrame successfully merged - {X_merged.shape}')
        merged_transactions_df = pd.read_csv(MERGED_DATASET_FILE, chunksize=CHUNK_SIZE, dtype=DTYPES_PREPROCESSED,  on_bad_lines='warn')

#######################################  FINAL PREPROCESSOR ####################################################################


    if merged_dataset_exists:
        print("📁 Loading Merged DataFrame iterable for processing from local CSV...")
        processed_chunks = []
        total_rows = sum(1 for _ in open(MERGED_DATASET_FILE)) - 1
        total_chunks = total_rows // CHUNK_SIZE

        for chunk_id, chunk in enumerate(merged_transactions_df):
            # Apply preprocessing to each chunk
            chunk_processed = post_merging_preprocessor(chunk)
            processed_chunks.append(chunk_processed)

            # Print progress every 10 iterations
            if (chunk_id + 1) % 10 == 0:
                print(f"Already preprocessed {chunk_id + 1} chunks ...")

        # Concatenate all processed chunks into a single DataFrame
        X_preprocessed = pd.concat(processed_chunks, axis=0, ignore_index=True)
        X_preprocessed.to_csv(FINAL_PROCESSED_DATASET_FILE, index=False)
        print(f'✅ Final DataFrame successfully processed - {X_preprocessed.shape}')
    else:
        raise Exception(Fore.RED + "⚠️ Please make sure the have correctly been merged before attempting to finalize the preprocessing." + Style.RESET_ALL)

    print("🎉 preprocessing() done")
    print(f'🎉 You can now move forward and train your desired model on the preprocessed dataset!')


def training(X_preprocessed: pd.DataFrame, model_type= MODEL_TYPE):

    X_preprocessed = X_preprocessed.copy()
    X_preprocessed = X_preprocessed.astype(DTYPES_PREPROCESSED)
    X_preprocessed.dropna(subset=['n_tax_households','average_tax_income'], axis=0, inplace=True)
    X_preprocessed = X_preprocessed.reset_index(drop=True)

    y = X_preprocessed.pop('price/m²')
    X = X_preprocessed

    X = X.sort_values(by='year_month_numeric', ascending=True)
    X_sorted = X.reset_index(drop=True)
    y_sorted = y.loc[X_sorted.index].reset_index(drop=True)

    # Splits the dataset into training and testing sets with a 70-30 ratio.
    # The split is performed in an ordered manner to preserve the temporal sequence of the data, ensuring that future data points do not appear in the training set.
    train_split_index = int(0.7 * len(X))

    X_train = X_sorted[:train_split_index]
    X_test = X_sorted[train_split_index:]
    y_train = y_sorted[:train_split_index]
    y_test = y_sorted[train_split_index:]

   # Further splits the training dataset into training and validation sets with a 80-20 ratio.
    val_split_index = int(0.8 * len(X_train))

    X_train_split = X_train[:val_split_index]
    X_val_split = X_train[val_split_index:]
    y_train_split = y_train[:val_split_index]
    y_val_split = y_train[val_split_index:]

    print(f"Training set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val_split)}")
    print(f"Testing set size: {len(X_test)}")

    if model_type == 'xgboost':

        model = None
        params = xgboost_model({
                                # "n_estimators": 50,
                                "learning_rate": 0.05,
                                "max_depth": 10,
                                "subsample": 0.9,
                                "tree_method": "hist",
                                "colsample_bytree": 0.9,
                                "objective": "reg:squarederror",
                                "random_state": 42,
                                "verbosity": 1,
                                'gamma': 0.1,
                                })

        print("✅ Model instantiated")

        model, metrics = train_xgb_model(params=params,
                                X = X_train_split, y = y_train_split,
                                X_val = X_val_split, y_val = y_val_split,
                                eval_metric="rmse",
                                early_stopping_rounds=15,  # Patience: stop after 15 rounds without improvement
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
        save_results(params=training_params, metrics=dict(rmse=val_rmse))

        # Save model weight on the hard drive (and optionally on GCS too)
        save_model(model=model)

        return val_rmse

    elif model_type == 'keras':

        model = None

        X_train_categorical = keras_preprocessor(X_train_split).values
        X_val_categorical = keras_preprocessor(X_val_split).values

        X_train_numeric= X_train_split.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values
        X_val_numeric= X_val_split.drop(columns=['departement', 'unique_city_id'], axis=1).astype(DTYPES_KERAS).values

        y_train = y_train_split
        y_val =  y_val_split

        X_departement = X_train_categorical[:, 0]
        X_unique_city = X_train_categorical[:, 1]

        X_departement_val = X_val_categorical[:, 0]
        X_unique_city_val = X_val_categorical[:, 1]

        learning_rate = 0.0001
        batch_size = 256
        patience=10

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
        save_results(params=params, metrics=dict(mae=val_rmse))
        save_model(model=model)

    else:
        raise Exception(Fore.RED + f"⚠️ Please select a valid model_type" + Style.RESET_ALL)


    print("🎉 training() done")
    print(f"🎉 {model_type} model has been successfully trained!")



if __name__ == '__main__':
    # clean_transactions_df, clean_new_mortgages_df, clean_tax_households_df, clean_interest_rates_df, clean_debt_ratio_df = cleaning_in_chunks()
    # preprocessing_in_chunks(clean_transactions_df, clean_new_mortgages_df, clean_tax_households_df, clean_interest_rates_df, clean_debt_ratio_df)
    training(pd.read_csv(FINAL_PROCESSED_DATASET_FILE))
