import os
import pandas as pd
import numpy as np
from real_estate_parquet.params import *

class real_estate:
    def get_data(self):
        """
        Retrieves data from the raw dataset folder, loading all Parquet files into a dictionary of pandas DataFrames.

        Returns:
            dict: A dictionary where keys are dataset names (e.g., 'flux_nouveaux_emprunts', 'foyers_fiscaux')
                  and values are pandas DataFrames containing the corresponding data from the Parquet files.
        """

        raw_data_path = os.path.join(RAW_DATASET_FOLDER)
        file_names = os.listdir(raw_data_path)
        suffix = '_df'

        parquet_dfs = []
        parquet_files = []

        data = {}

        for file in file_names:
            if file.endswith('.parquet'):
                parquet_files.append(file)
                file = file.replace('.parquet', suffix)
                parquet_dfs.append(file)

                for files, paths in zip(parquet_dfs, parquet_files):
                    data[files] = pd.read_parquet(os.path.join(raw_data_path, paths), engine='pyarrow')

        return data


def save_data_to_parquet(raw_data_folder = RAW_DATASET_FOLDER):
    """
    Converts CSV files in the specified directory to Parquet format and returns a dictionary of the loaded DataFrames.

    The dictionary's keys are the names of the datasets (after removing the '.csv' extension), and the values
    are pandas DataFrames created from the corresponding CSV files.

    Args:
        raw_data_folder (str): The directory containing the raw CSV files to be converted and saved as Parquet. Defaults to `RAW_DATASET_FOLDER`.

    Returns:
        dict: A dictionary containing the CSV files as pandas DataFrames, with keys being the file names without extensions.
    """

    # List all files in the directory
    file_names = os.listdir(raw_data_folder)

    # Remove specific files/directories that should not be processed
    for file_to_remove in ['transactions_sample.csv', 'transactions.npz', 'raw_dataset_chunks', 'raw_dataset_full']:
        if file_to_remove in file_names:
            file_names.remove(file_to_remove)

    # Dictionary to store DataFrames
    data = {}

    # Convert each CSV file to a DataFrame and save as Parquet
    for file_name in file_names:
        csv_path = os.path.join(raw_data_folder, file_name)
        parquet_path = os.path.join(raw_data_folder, file_name.replace('.csv', '.parquet'))

        df = pd.read_csv(csv_path)
        df.to_parquet(parquet_path, engine='pyarrow', index=False)

        print(f"Converted {csv_path} to {parquet_path}")

        # Store in dictionary with cleaned key name
        key = file_name.replace('.csv', '')  # Remove .csv from key
        data[key] = df


def clean_transactions(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the transactions data by removing irrelevant columns, handling missing values, and transforming data.

    Specifically, it:
    - Drops unnecessary columns.
    - Removes duplicates.
    - Converts the 'date_transaction' column to datetime format.
    - Filters out rows with agricultural or industrial land.
    - Cleans and converts the 'surface_terrains_sols' column.
    - Filters the dataset based on specific conditions (e.g., only metropolitan France departments).
    - Categorizes 'surface_terrains_sols' into discrete bins.

    Args:
        X (pd.DataFrame): The raw transactions DataFrame to be cleaned.

    Returns:
        pd.DataFrame: A cleaned DataFrame with relevant columns, properly formatted dates, and discretized surface areas.
    """

    assert isinstance(X, pd.DataFrame)

    # Only keeping relevant columns for our model
    columns_to_drop = ['id_transaction', 'code_postal', 'adresse', 'vefa', 'id_parcelle_cadastre',
                    'surface_dependances', 'surface_terrains_nature', 'latitude', 'longitude']
    X = X.drop(columns=columns_to_drop, axis=1)

    # Dropping duplicates while keeping first occurence
    X = X.drop_duplicates(ignore_index=True)

    # Ensure datetime column is properly parsed
    X["date_transaction"] = pd.to_datetime(X["date_transaction"], errors="coerce")

    # Dropping the rows where the outdoor area is either an industrial surface or an agricultural land plot. We are only going to focus on the private properties.
    mask = (X['surface_terrains_agricoles'] != '{}') | (X['surface_locaux_industriels'] != '{}')
    X = X[~mask]

    # Cleaning Surface of outbuildings column before processing
    X['surface_terrains_sols'] = X.surface_terrains_sols.replace(r'[{}]', '', regex=True)
    X['surface_terrains_sols'] = X['surface_terrains_sols'].apply(lambda x: 0 if ',' in str(x) else x)

    # Casting surface_terrains_sols to_numeric.
    X['surface_terrains_sols']  = pd.to_numeric(X['surface_terrains_sols'], errors='coerce')

    # Replacing np.nan by 0 in Surface of outbuildings.
    X['surface_terrains_sols'] = X['surface_terrains_sols'].replace(np.nan, 0)

    X = X.drop(columns=['surface_locaux_industriels', 'surface_terrains_agricoles'])

    X = X.astype(DTYPES_RAW)

    # Dropping rows with NaN values resulting from coercion or filtering
    X = X.dropna()

    # Only keeping departements of Metropolitain France
    X = X[X.departement <= 95]

    # Keeping rows where the #rooms is higher than 0
    X = X[X.n_pieces > 0]

    # Discretizing surface of outbuildings into 4 distinct categories.
    bins = [-1, 0, 200, 1000, float('inf')]
    labels = ['No garden', 'Small outdoor space', 'Average outdoor space', 'Large outdoor space']
    X['surface_terrains_sols'] = pd.cut(X['surface_terrains_sols'], bins=bins, labels=labels)

    X = X.astype(DTYPES_CLEANED)

    return X

def clean_data():
    def clean_tax_households(X: pd.DataFrame, city_mapping_path = CITY_MAPPING_PATH) -> pd.DataFrame:
        """
        Cleans multiple datasets related to tax households, debt ratios, interest rates, and new mortgages.

        The function applies specific cleaning procedures to each dataset, such as:
        - Merging with city mappings for tax household data.
        - Resampling and interpolating debt ratio data.
        - Sorting and cleaning interest rates data.
        - Processing new mortgages data by converting units.

        Returns:
            dict: A dictionary where keys are dataset names (e.g., 'foyers_fiscaux_df', 'taux_endettement_df') and values are the cleaned DataFrames.
        """

        city_mapping = pd.read_parquet(city_mapping_path)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(city_mapping, pd.DataFrame)

        X = X.copy()
        X = X[['date', 'departement', 'ville', 'n_foyers_fiscaux', 'revenu_fiscal_moyen']]

        X['ville'] = X['ville'].apply(lambda row: row.upper())

        X_merged = pd.merge(left=X, right=city_mapping, on='ville')
        X_merged.drop(columns=['date', 'departement', 'ville'], axis =1, inplace=True)

        X_merged = X_merged.groupby(['unique_city_id']).agg({
                                                            'n_foyers_fiscaux': 'mean',
                                                            'revenu_fiscal_moyen': 'mean'
                                                            }).reset_index()

        X_merged = X_merged.astype({"unique_city_id": "string",
                                   "n_foyers_fiscaux": "float64",
                                   "revenu_fiscal_moyen": "float64"})


        print(f'✅ Tax households DataFrame cleaned - {X_merged.shape}')

        return X_merged


    def clean_debt_ratio(X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)

        X = X.copy()

        X['date'] = pd.to_datetime(X['date'], format='%Y')
        X.set_index('date', inplace=True)
        X = X.resample('MS').interpolate(method='linear')

        print(f'✅ Debt ratio DataFrame cleaned - {X.shape}')

        return X


    def clean_interest_rates(X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)

        X = X.copy()

        X = X.set_index(X['date']).drop('date', axis=1)
        X.index = pd.to_datetime(X.index)
        X = X.sort_index(axis = 0, ascending=True)

        print(f'✅ Interest rates DataFrame cleaned - {X.shape}')

        return X


    def clean_new_mortgages(X: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(X, pd.DataFrame)

        X = X.copy()

        X['emprunts_€'] = X['emprunts_M€'] * 1000000
        X.drop('emprunts_M€', axis = 1, inplace=True)
        X = X.set_index(X['date']).sort_index(axis=0, ascending=True).drop('date', axis=1)
        X.index = pd.to_datetime(X.index)

        print(f'✅ New mortgages DataFrame cleaned - {X.shape}')

        return X

    ##################  DATA EXTRACTION & PROCESSING PATHS  #####################

    sorted_names_of_dataframes = ['foyers_fiscaux_df',
                            'taux_interet_df',
                            'flux_nouveaux_emprunts_df',
                            'taux_endettement_df']

    sorted_names_of_cleaning_functions = [clean_tax_households,
                                   clean_interest_rates,
                                   clean_new_mortgages,
                                   clean_debt_ratio]

    dataframes_dictionnary = {}
    for name in sorted_names_of_dataframes:
        dataframes_dictionnary[name] = real_estate().get_data()[name].copy()

    cleaned_dataframes_dictionnary = {}
    for df_name, cleaning_function in zip(sorted_names_of_dataframes, sorted_names_of_cleaning_functions):
        # Apply the cleaning function to the DataFrame
        cleaned_df = cleaning_function(dataframes_dictionnary[df_name])
        cleaned_df.to_parquet(os.path.join(CLEANED_DATASET_FOLDER, 'cleaned_' + df_name + '.parquet'), index=False, engine='pyarrow', compression ='snappy')
        cleaned_dataframes_dictionnary[df_name] = cleaned_df

    return cleaned_dataframes_dictionnary


def combined_temporal_features_df(clean_new_mortgages: pd.DataFrame, clean_interest_rates: pd.DataFrame,
                    clean_debt_ratio: pd.DataFrame) -> pd.DataFrame:
    """
    Combines multiple temporal datasets (new mortgages, interest rates, and debt ratios) into a single DataFrame.

    The function ensures that all input DataFrames are reindexed to the same monthly frequency and interpolates any missing values.
    It then combines these datasets into a single DataFrame with temporal features.

    Args:
        clean_new_mortgages (pd.DataFrame): Cleaned new mortgages data with a datetime index.
        clean_interest_rates (pd.DataFrame): Cleaned interest rates data with a datetime index.
        clean_debt_ratio (pd.DataFrame): Cleaned debt ratio data with a datetime index.

    Returns:
        pd.DataFrame: A DataFrame containing merged temporal features, with a 'year_month_numeric' column and the respective financial data.
    """

    assert isinstance(clean_new_mortgages, pd.DataFrame)
    assert isinstance(clean_interest_rates, pd.DataFrame)
    assert isinstance(clean_debt_ratio, pd.DataFrame)

    # Reindex all input DataFrames to the same monthly frequency
    common_index = pd.date_range(start=clean_new_mortgages.index[0],
                                    end=clean_new_mortgages.index[-1],
                                    freq='MS')

    clean_new_mortgages = clean_new_mortgages.reindex(common_index).interpolate()
    clean_debt_ratio = clean_debt_ratio.reindex(common_index).interpolate()
    clean_interest_rates = clean_interest_rates.reindex(common_index).interpolate()

    # Combine into a single DataFrame
    combined_temporal_features = pd.DataFrame({
            'New_mortgages': clean_new_mortgages.squeeze(),
            'Debt_ratio': clean_debt_ratio.squeeze(),
            'Interest_rates': clean_interest_rates.squeeze()
        }, index=common_index)

    combined_temporal_features.dropna(how='any', inplace=True)
    combined_temporal_features['date'] = combined_temporal_features.index
    combined_temporal_features['year_month_numeric'] = combined_temporal_features['date'].dt.year * 12 + combined_temporal_features['date'].dt.month
    combined_temporal_features.drop('date', axis=1, inplace=True)

    combined_temporal_features = combined_temporal_features.astype({"year_month_numeric": "int32",
                                                                    "New_mortgages": "float64",
                                                                    "Debt_ratio": "float32",
                                                                    "Interest_rates": "float32"})

    print(f'✅ Temporal features successfully merged - {combined_temporal_features.shape}')

    return combined_temporal_features


def merged_dfs(preprocessed_transactions_df, temporal_features_df: pd.DataFrame,
                    tax_households_df: pd.DataFrame, primary_keys = ('year_month_numeric', 'unique_city_id')) -> pd.DataFrame:
    assert isinstance(temporal_features_df, pd.DataFrame)
    assert isinstance(tax_households_df, pd.DataFrame)

    """
    Merges the preprocessed transaction data with temporal features and tax household data based on common keys.

    Args:
        preprocessed_transactions_df (pd.DataFrame): The preprocessed transactions DataFrame.
        temporal_features_df (pd.DataFrame): The temporal features DataFrame (e.g., new mortgages, interest rates).
        tax_households_df (pd.DataFrame): The tax households data.
        primary_keys (tuple): A tuple of column names to be used as keys for merging the DataFrames. Defaults to ('year_month_numeric', 'unique_city_id').

    Returns:
        pd.DataFrame: A DataFrame containing the merged data with the specified primary keys.
    """

    X_merged =  preprocessed_transactions_df.merge(temporal_features_df, how='inner', on=primary_keys[0], copy=True)
    X_merged = X_merged.merge(tax_households_df, on=primary_keys[1], how='inner', copy=True)

    return X_merged
