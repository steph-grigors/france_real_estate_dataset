import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def transform_date_transactions(X: pd.DataFrame) -> np.ndarray:
    """
    Transforms the 'date_transaction' column in the input DataFrame into numerical and cyclic encoded features.

    Specifically, it:
    - Extracts the year and month from the 'date_transaction' column.
    - Creates a numerical encoding by combining the year and month.
    - Applies cyclic encoding for the month using sine and cosine transformations.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'date_transaction' column.

    Returns:
        pd.DataFrame: A DataFrame containing the numerical and cyclic features: 'year_month_numeric', 'month_sin', and 'month_cos'.
    """

    assert isinstance(X, pd.DataFrame)

    transactions = pd.to_datetime(X['date_transaction'])
    year = transactions.dt.year
    month = transactions.dt.month

    # Numerical encoding: year * 12 + month
    year_month_numeric = year * 12 + month

    # Cyclic encoding for the month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return pd.DataFrame({'year_month_numeric': year_month_numeric,
                         'month_sin': month_sin,
                         'month_cos':month_cos})


def transform_target(X: pd.DataFrame) -> pd.Series:
    """
    Transforms the target variable by calculating the price per square meter and applying a logarithmic transformation.

    Specifically, it:
    - Computes the price per square meter by dividing 'prix' by 'surface_habitable'.
    - Replaces 0 and NaN values with a small constant (1e-5).
    - Returns the logarithm of the price per square meter and the 'surface_habitable' column.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'prix' and 'surface_habitable' columns.

    Returns:
        pd.DataFrame: A DataFrame with the logarithmic price per square meter ('log_price_per_m2') and the 'surface_habitable' column.
    """

    assert isinstance(X, pd.DataFrame)

    price_per_sqm = X['prix']/X['surface_habitable']
    price_per_sqm = price_per_sqm.replace({0: 1e-5, np.nan: 1e-5})

    return pd.DataFrame({'log_price_per_m2': np.log(price_per_sqm),
                         'surface_habitable': X['surface_habitable']}
                        )


def feature_engineer_unique_city(X: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a unique city identifier by combining the 'departement' and 'id_ville' columns.

    Specifically, it:
    - Constructs the 'unique_city_id' by combining 'departement' and 'id_ville' into a tuple string.
    - Returns a DataFrame with the 'departement' and 'unique_city_id' columns.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'departement' and 'id_ville' columns.

    Returns:
        pd.DataFrame: A DataFrame containing the 'departement' and the newly created 'unique_city_id'.
    """

    assert isinstance(X, pd.DataFrame)

    X['unique_city_id'] = X.apply(lambda row: f"({row['departement']},{row['id_ville']})", axis=1)

    return pd.DataFrame(X[['departement', 'unique_city_id']])


def keras_departement_encoder(X: pd.DataFrame):
    """
    Encodes the 'departement' column in the input DataFrame using a label encoder.

    Specifically, it:
    - Applies a LabelEncoder to the 'departement' column.
    - Returns the transformed 'departement' column as a reshaped numpy array.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'departement' column.

    Returns:
        np.ndarray: A numpy array containing the encoded 'departement' values, reshaped for compatibility with Keras.
    """
    dept_encoder = LabelEncoder()
    X['departement'] = dept_encoder.fit_transform(X['departement'])
    return X['departement'].values.reshape(-1, 1)

def keras_unique_city_id_encoder(X: pd.DataFrame):
    """
    Encodes the 'unique_city_id' column in the input DataFrame using a label encoder.

    Specifically, it:
    - Applies a LabelEncoder to the 'unique_city_id' column.
    - Returns the transformed 'unique_city_id' column as a reshaped numpy array.

    Args:
        X (pd.DataFrame): The input DataFrame containing the 'unique_city_id' column.

    Returns:
        np.ndarray: A numpy array containing the encoded 'unique_city_id' values, reshaped for compatibility with Keras.
    """
    city_encoder = LabelEncoder()
    X['unique_city_id'] = city_encoder.fit_transform(X['unique_city_id'])
    return X['unique_city_id'].values.reshape(-1, 1)
