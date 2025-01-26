import numpy as np
import pandas as pd

from colorama import Fore, Style

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler


from real_estate.ml_logic.encoders import *


def preprocess_features(X: pd.DataFrame) -> np.ndarray:
    def preprocessor_transactions() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned dataset of shape (_, 7)
        into a preprocessed one of fixed shape (_, 65).

        Stateless operation: "fit_transform()" equals "transform()".
        """

        # NUMBER OF ROOMS PIPE
        n_rooms_pipe = make_pipeline(MinMaxScaler())

        # PRICE PIPE
        price_pipe = make_pipeline(
            FunctionTransformer(transform_target)
        )

        # UNIQUE CITY PIPE
        unique_city_id_pipe = make_pipeline(
            FunctionTransformer(feature_engineer_unique_city)
        )

        # TIME PIPE
        time_pipe = make_pipeline(
            FunctionTransformer(transform_date_transactions)
        )

        # CAT FEATURES OHE
        OHE = OneHotEncoder(drop='if_binary',
                            sparse_output=False,
                            handle_unknown="ignore")


        # NUM FEATURES
        living_area_pipe = make_pipeline(StandardScaler())


        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("n_rooms_normalizer", n_rooms_pipe, ['n_pieces']),
                ("time_preproc", time_pipe, ['date_transaction']),
                ("unique_city_preproc", unique_city_id_pipe, ['departement', 'id_ville', 'ville']),
                ("building_type", OHE, ['type_batiment', 'surface_terrains_sols']),
                ("price_preproc", price_pipe, ['prix', 'surface_habitable']),
                ('living_area_preproc', living_area_pipe, ['surface_habitable'])
            ],
            n_jobs=-1,
        )

        return final_preprocessor


    print(Fore.BLUE + "\nPreprocessing features from raw transactions dataframe..." + Style.RESET_ALL)

    preprocessor = preprocessor_transactions()
    X_transactions_processed = pd.DataFrame(preprocessor.fit_transform(X))
    X_transactions_processed.columns = ['n_rooms', 'year_month_numeric', 'month_sin', 'month_cos',
                           'departement', 'unique_city_id',
                           'no_garden', 'small_outdoor_space', 'average_outdoor_space', 'large_outdoor_space',
                           'building_type','price/m²', 'living_area']


    # print("✅ X_processed, with shape", X_transactions_processed.shape)

    return X_transactions_processed

def final_preprocessor(X_transactions_processed) -> ColumnTransformer:
    def preprocessor_processed_transactions() -> ColumnTransformer:

        tax_households_pipeline = make_pipeline(StandardScaler())
        temporal_features_pipeline = make_pipeline(MinMaxScaler())

        final_preprocessor = make_column_transformer(
            (tax_households_pipeline, ['n_foyers_fiscaux', 'revenu_fiscal_moyen']),
            (temporal_features_pipeline, ['New_mortgages', 'Debt_ratio', 'Interest_rates']),
            remainder='passthrough',
            n_jobs=-1
        )

        return final_preprocessor

    preprocessor = preprocessor_processed_transactions()
    X_transactions_processed = pd.DataFrame(preprocessor.fit_transform(X_transactions_processed))
    X_transactions_processed.columns = ['n_tax_households', 'average_tax_income',
                                        'new_mortgages', 'debt_ratio',
                                        'interest_rates', 'n_rooms', 'year_month_numeric', 'month_sin', 'month_cos',
                                            'departement', 'unique_city_id',
                                            'no_garden', 'small_outdoor_space', 'average_outdoor_space', 'large_outdoor_space',
                                            'building_type','price/m²', 'living_area']


    # ['n_rooms', 'year_month_numeric', 'month_sin', 'month_cos',
    #                        'departement', 'unique_city_id',
    #                        'no_garden', 'small_outdoor_space', 'average_outdoor_space', 'large_outdoor_space',
    #                        'building_type','price/m²', 'living_area', 'new_mortgages',	'debt_ratio',
    #                        'interest_rates', 'n_tax_households', 'average_tax_income']

    print("✅ X_processed, with shape", X_transactions_processed.shape)

    return X_transactions_processed
