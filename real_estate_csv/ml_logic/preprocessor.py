import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MinMaxScaler


from real_estate_csv.ml_logic.encoders import *
from real_estate_csv.params import *


def stateless_preprocessor(X: pd.DataFrame) -> pd.DataFrame:

    # Drop 'ville' column from the original DataFrame
    X = X.drop(columns=['ville'], axis=1)

    def preprocessor() -> ColumnTransformer:
        """
        Scikit-learn pipeline that transforms a cleaned chunk
        into a preprocessed one.
        Stateless operation: "fit_transform()" equals "transform()".
        """

        # PRICE PIPE
        price_pipe = make_pipeline(
            FunctionTransformer(transform_target, validate=False)
        )

        # UNIQUE CITY PIPE
        unique_city_id_pipe = make_pipeline(
            FunctionTransformer(feature_engineer_unique_city, validate=False)
        )

        # TIME PIPE
        time_pipe = make_pipeline(
            FunctionTransformer(transform_date_transactions, validate=False)
        )

        # COMBINED PREPROCESSOR
        final_preprocessor = ColumnTransformer(
            [
                ("time_preproc", time_pipe, ['date_transaction']),
                ("unique_city_preproc", unique_city_id_pipe, ['departement', 'id_ville']),
                ("price_preproc", price_pipe, ['prix', 'surface_habitable']),
            ],
            remainder="passthrough",
            n_jobs=-1,
        )

        return final_preprocessor

    # Fit and transform X
    preprocessor = preprocessor()
    X_processed = preprocessor.fit_transform(X)

    # Get column names for the transformed features
    transformed_feature_names = (
        ["year_month_numeric", "month_sin", "month_cos"] +  # Time pipeline
        ["departement", "unique_city_id"] +  # Unique city pipeline
        ["log_price_per_m2", "living_area"] +  # Price pipeline
        [col for col in X.columns if col not in ['date_transaction', 'departement', 'id_ville', 'prix', "surface_habitable"]]  # Keep other columns
    )

    # Ensure transformed columns match X_processed shape
    X_transactions_processed = pd.DataFrame(X_processed, columns=transformed_feature_names, index=X.index)

    return X_transactions_processed.astype(DTYPES_STATELESS_PROCESSED)


def post_merging_preprocessor(X: pd.DataFrame, preprocessor = None, fit: bool = True) -> np.ndarray:
    """
    Returns a fitted preprocessor for the training set or a preprocessor ready to transform the test set.
    """
    def build_preprocessor() -> ColumnTransformer:
        """
        Statelful operation: "fit_transform()" does not equal "transform()".
        """

        # NUMBER OF ROOMS + TEMPORAL FEATURES PIPES - MINMAX
        numerical_features_pipe = make_pipeline(MinMaxScaler())

        # CAT FEATURES OHE
        cat_pipe_ohe = make_pipeline(OneHotEncoder(drop='if_binary',
                            sparse_output=False,
                            handle_unknown="ignore")
                            )

        # LIVING AREA + TAX JOUSEHOLDS PIPE - STD SCALER
        num_features_sc_pipe = make_pipeline(StandardScaler())


        # COMBINED PREPROCESSOR
        preprocessor = ColumnTransformer(
            [
                ("normalizer", numerical_features_pipe, ['n_rooms', 'new_mortgages','debt_ratio','interest_rates']),
                ("ohe_cat", cat_pipe_ohe, ['building_type', 'outdoor_area']),
                ('standardizer', num_features_sc_pipe, ['living_area', 'n_tax_households', 'average_tax_income'])
            ],
            n_jobs=-1,
            remainder="passthrough",
        )

        return preprocessor

    # Use provided preprocessor if given, otherwise create a new one
    if preprocessor is None:
            preprocessor = build_preprocessor()

    if fit:
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)



    # Get column names for the transformed features
    transformed_feature_names = (
            ['n_rooms', 'new_mortgages','debt_ratio','interest_rates'] +  # NUMBER OF ROOMS + TEMPORAL FEATURES PIPES - MINMAX
            ['building_type', 'average_outdoor_space', 'large_outdoor_space', 'no_garden', 'small_outdoor_space'] +  # CAT FEATURES OHE
            ['living_area', 'n_tax_households', 'average_tax_income'] +  # LIVING AREA + TAX HOUSEHOLDS PIPE - STD SCALER
            [col for col in X.columns if col not in ['n_rooms', 'new_mortgages', 'debt_ratio', 'interest_rates',
                                            'building_type', 'surface_terrains_sols',
                                            'living_area', 'n_tax_households', 'average_tax_income', 'outdoor_area']]
    )

    X_merged_processed = pd.DataFrame(X_processed, columns=transformed_feature_names, index=X.index)

    return X_merged_processed.astype(DTYPES_PREPROCESSED), preprocessor


def keras_preprocessor(X_preprocessed) ->  np.ndarray:
    def preprocessor() -> ColumnTransformer:

        departement_pipeline = make_pipeline(
            FunctionTransformer(keras_departement_encoder)
        )
        unique_city_id_pipeline = make_pipeline(
            FunctionTransformer(keras_unique_city_id_encoder)
        )

        final_preprocessor = make_column_transformer(
            (departement_pipeline, ['departement']),
            (unique_city_id_pipeline, ['unique_city_id']),
            remainder='drop',
            n_jobs=-1
        )

        return final_preprocessor

    preprocessor = preprocessor()
    X_preprocessed = pd.DataFrame(preprocessor.fit_transform(X_preprocessed))
    X_preprocessed.columns = ['departement', 'unique_city_id']

    return X_preprocessed
