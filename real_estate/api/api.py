import pandas as pd
import numpy as np
import joblib

from real_estate.ml_logic.registry import load_model
from real_estate.ml_logic.preprocessor import post_merging_preprocessor, keras_preprocessor
from real_estate.params import *

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 💡 Preload the model to accelerate the predictions
app.state.model = load_model(model_type=MODEL_TYPE)

@app.get("/predict")
def predict(
    year_month_numeric: int,
    month_sin: float,
    month_cos: float,
    departement: str,
    unique_city_id: str,
    living_area: float,
    building_type: str,
    n_rooms: int,
    outdoor_area: str,
    new_mortgages: float,
    debt_ratio: float,
    interest_rates: float,
    n_tax_households: float,
    average_tax_income: float):

    X_pred = pd.DataFrame(locals(), index=[0])

    # Ensure model is loaded
    if app.state.model is None:
        print("⚠️ Model was not loaded at startup, reloading now...")
        app.state.model = load_model(model_type=MODEL_TYPE)

    model = app.state.model
    assert model is not None, "❌ Model failed to load"

    fitted_preprocessor = joblib.load(PREPROCESSOR_PATH)
    X_pred_processed, _ = post_merging_preprocessor(X_pred, preprocessor=fitted_preprocessor, fit=False)

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
    return dict(price_per_sq_meter = float(y_pred))


@app.get("/")
def root():
    pass

if __name__ == "__main__":
    predict(24260,-0.8660254037844384,-0.5000000000000004,27,"(27, 185)",67.0,"Maison",2,"No garden",26094000000.0,102.92333,1.09,263.55555555555554,28603.32777777778
)
