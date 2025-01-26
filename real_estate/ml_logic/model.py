import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple

import xgboost as xgb
from sklearn.dummy import DummyRegressor
# from tensorflow import keras
# from keras import Model, Sequential, layers, regularizers, optimizers
# from keras.callbacks import EarlyStopping


def baseline_model():
 dummy_model = DummyRegressor(strategy='mean')

def xgboost_model(params):

    """
    Create a dictionary of hyperparameters for the XGBoost model.
    """
    # Parameters should be passed as a dictionary
    default_params = {
        "n_estimators": params.get("n_estimators", 100),
        "learning_rate": params.get("learning_rate", 0.1),
        "max_depth": params.get("max_depth", 6),
        "min_child_weight": params.get("min_child_weight", 1),
        "subsample": params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "gamma": params.get("gamma", 0),
        "reg_alpha": params.get("reg_alpha", 0),
        "reg_lambda": params.get("reg_lambda", 1),
        "objective": params.get("objective", "reg:squarederror"),
        "random_state": params.get("random_state", 42),
        "verbosity": params.get("verbosity", 1),
    }
    return default_params

def train_xgb_model(params, X, y, X_val=None, y_val=None, eval_metric="rmse", early_stopping_rounds=15, verbose=True):
    """
    Train an XGBoost model using DMatrix with categorical data handling.
    """
    # Create DMatrices for training and validation sets
    dtrain = xgb.DMatrix(X, label=y, enable_categorical=True)
    evals = [(dtrain, "train")]

    # Add validation set if provided
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
        evals.append((dval, "validation"))

    # Train the model
    evals_result = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=params.get("n_estimators", 100),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
        evals_result=evals_result,  # Optionally track evaluation metrics during training
    )

    return model, evals_result


# def neural_network_model():
#     def initialize_model(input_shape: tuple) -> Model:
#         """
#         Initialize the Neural Network with random weights
#         """
#         reg = regularizers.l1_l2(l2=0.005)

#         model = Sequential()
#         model.add(layers.Input(shape=input_shape))
#         model.add(layers.Dense(100, activation="relu", kernel_regularizer=reg))
#         model.add(layers.BatchNormalization(momentum=0.9))
#         model.add(layers.Dropout(rate=0.1))
#         model.add(layers.Dense(50, activation="relu"))
#         model.add(layers.BatchNormalization(momentum=0.9))  # use momentum=0 to only use statistic of the last seen minibatch in inference mode ("short memory"). Use 1 to average statistics of all seen batch during training histories.
#         model.add(layers.Dropout(rate=0.1))
#         model.add(layers.Dense(1, activation="linear"))

#         print("✅ Model initialized")

#         return model


#     def compile_model(model: Model, learning_rate=0.0005) -> Model:
#         """
#         Compile the Neural Network
#         """
#         optimizer = optimizers.Adam(learning_rate=learning_rate)
#         model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["mae"])

#         print("✅ Model compiled")

#         return model

#     def train_model(
#             model: Model,
#             X: np.ndarray,
#             y: np.ndarray,
#             batch_size=256,
#             patience=2,
#             validation_data=None, # overrides validation_split
#             validation_split=0.3
#         ) -> Tuple[Model, dict]:
#         """
#         Fit the model and return a tuple (fitted_model, history)
#         """
#         print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

#         es = EarlyStopping(
#             monitor="val_loss",
#             patience=patience,
#             restore_best_weights=True,
#             verbose=1
#         )

#         history = model.fit(
#             X,
#             y,
#             validation_data=validation_data,
#             validation_split=validation_split,
#             epochs=100,
#             batch_size=batch_size,
#             callbacks=[es],
#             verbose=0
#         )

#         print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

#         return model, history


#     def evaluate_model(
#             model: Model,
#             X: np.ndarray,
#             y: np.ndarray,
#             batch_size=64
#         ) -> Tuple[Model, dict]:
#         """
#         Evaluate trained model performance on the dataset
#         """

#         print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

#         if model is None:
#             print(f"\n❌ No model to evaluate")
#             return None

#         metrics = model.evaluate(
#             x=X,
#             y=y,
#             batch_size=batch_size,
#             verbose=0,
#             # callbacks=None,
#             return_dict=True
#         )

#         loss = metrics["loss"]
#         mae = metrics["mae"]

#         print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

#         return metrics
