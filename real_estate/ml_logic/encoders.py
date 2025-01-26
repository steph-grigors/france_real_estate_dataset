import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



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

    return pd.DataFrame({'price/m²': price_per_sqm
                         })


def feature_engineer_unique_city(X: pd.DataFrame, cache_path='city_mapping.csv') -> pd.DataFrame:
    assert isinstance(X, pd.DataFrame)

    X['unique_city_id'] = X.apply(lambda row: (row['departement'], row['id_ville']), axis=1)

    return pd.DataFrame(X[['departement', 'unique_city_id']])
