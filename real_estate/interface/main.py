import numpy as np
import pandas as pd
import ipdb

from pathlib import Path
from colorama import Fore, Style

from real_estate.ml_logic import *
from real_estate.ml_logic.data_extraction import *
from real_estate.ml_logic.data import *
from real_estate.ml_logic.preprocessor import *
from real_estate.ml_logic.model import xgboost_model, train_xgb_model
from real_estate.ml_logic.registry import save_model, save_results

from real_estate.params import *
from real_estate.utils import *



def data_extraction():

    # Extract the transactions by chunk from the .npz file and save them in an output_dir
    process_npz_chunkwise_to_csv(NPZ_FILE_PATH, CHUNK_SIZE, OUTPUT_DIR)

    # Optional step - concatenate the processed chunks into a single .csv file
    concatenate_csv_files(OUTPUT_DIR, OUTPUT_FILE)


def preprocessing():

    # Clean data using data.py
    cleaned_dataframes_dictionnary = clean_data()

    clean_transactions_df = cleaned_dataframes_dictionnary['transactions_sample_df']
    clean_new_mortgages_df = cleaned_dataframes_dictionnary['flux_nouveaux_emprunts_df']
    clean_tax_households_df = cleaned_dataframes_dictionnary['foyers_fiscaux_df']
    clean_interest_rates_df = cleaned_dataframes_dictionnary['taux_interet_df']
    clean_debt_ratio_df = cleaned_dataframes_dictionnary['taux_endettement_df']

    #
    create_city_mapping(clean_transactions_df)

    #
    temporal_features_df = combined_temporal_features_df(clean_new_mortgages_df,
                                                         clean_interest_rates_df,
                                                         clean_debt_ratio_df)

    ###### CODE
    clean_transactions_df_processed = preprocess_features(clean_transactions_df)

    #
    merged_dataframe = merged_dfs(clean_transactions_df_processed,
                                  temporal_features_df,
                                  clean_tax_households_df,
                                  primary_keys = ('year_month_numeric', 'unique_city_id'))

    processed_data_final = final_preprocessor(merged_dataframe)

    X = processed_data_final.drop('price/m²', axis=1)
    y = processed_data_final['price/m²']

    numeric_columns = [
    'n_tax_households', 'average_tax_income', 'new_mortgages',
    'debt_ratio', 'interest_rates', 'n_rooms', 'year_month_numeric',
    'month_sin', 'month_cos', 'living_area'
    ]
    X[numeric_columns] = X[numeric_columns].apply(pd.to_numeric, errors='coerce')

    categorical_columns = ['departement', 'unique_city_id', 'no_garden',
    'small_outdoor_space', 'average_outdoor_space',
    'large_outdoor_space', 'building_type'
    ]

    X[categorical_columns] = X[categorical_columns].astype('category')

    y = y.apply(pd.to_numeric, errors='coerce')

    print("✅ preprocess() done \n")

    return X, y

def training(X, y):

    model = None

    X = X.sort_values(by='year_month_numeric', ascending=True)
    X_sorted = X.reset_index(drop=True)
    y_sorted = y.loc[X_sorted.index].reset_index(drop=True)

    split_value = 24277

    X_train = X_sorted[X_sorted['year_month_numeric'] < split_value].copy()
    X_test = X_sorted[X_sorted['year_month_numeric'] >= split_value].copy()
    y_train = y_sorted[X_sorted['year_month_numeric'] < split_value].copy()
    y_test = y_sorted[X_sorted['year_month_numeric'] >= split_value].copy()

    # First 80% of the data is for training, the remaining 20% for validation
    val_split_index = int(0.8 * len(X_train))

    X_train_split = X_train[:val_split_index]
    X_val_split = X_train[val_split_index:]
    y_train_split = y_train[:val_split_index]
    y_val_split = y_train[val_split_index:]

    params = xgboost_model({
                            "n_estimators": 200,
                            "learning_rate": 0.05,
                            "max_depth": 8,
                            "subsample": 0.8,
                            "colsample_bytree": 0.8,
                            "objective": "reg:squarederror",
                            "random_state": 42,
                            "verbosity": 1,
                            })

    model, metrics = train_xgb_model(params=params,
                            X = X_train_split, y = y_train_split,
                            X_val = X_val_split, y_val = y_val_split,
                            eval_metric="rmse",
                            early_stopping_rounds=15,  # Patience: stop after 15 rounds without improvement
                            verbose=True,
                            )

    val_rmse = np.min(metrics['validation']['rmse'])

    training_params = dict(
        context="train",
        training_set_size=DATA_SIZE,
        row_count=len(X_train),
    )

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(params=training_params, metrics=dict(rmse=val_rmse))

    # Save model weight on the hard drive (and optionally on GCS too!)
    save_model(model=model)


    print(f"Training set size: {len(X_train)}")
    print(f"Testing set size: {len(X_test)}")

    return val_rmse


if __name__ == '__main__':
    X, y = preprocessing()
    training(X, y)
