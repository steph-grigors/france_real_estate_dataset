import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder




def transform_date_transactions(X: pd.DataFrame) -> np.ndarray:
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
    assert isinstance(X, pd.DataFrame)

    price_per_sqm = X['prix']/X['surface_habitable']
    price_per_sqm = price_per_sqm.replace({0: 1e-5, np.nan: 1e-5})

    return pd.DataFrame({'log_price/m²': np.log(price_per_sqm),
                         'surface_habitable': X['surface_habitable']}
                        )


def feature_engineer_unique_city(X: pd.DataFrame) -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)

    X['unique_city_id'] = X.apply(lambda row: (int(row['departement']), int(row['id_ville'])), axis=1)

    return pd.DataFrame(X[['departement', 'unique_city_id']])


def keras_departement_encoder(X: pd.DataFrame):
# Encode 'departement'
    dept_encoder = LabelEncoder()
    X['departement'] = dept_encoder.fit_transform(X['departement'])
    return X['departement'].values.reshape(-1, 1)

def keras_unique_city_id_encoder(X: pd.DataFrame):
    # Encode 'unique_city_id'
    city_encoder = LabelEncoder()
    X['unique_city_id'] = city_encoder.fit_transform(X['unique_city_id'])
    return X['unique_city_id'].values.reshape(-1, 1)
