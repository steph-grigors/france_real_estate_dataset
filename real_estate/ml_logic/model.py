import numpy as np
from typing import Tuple
from colorama import Fore, Style

import tensorflow as tf
import xgboost as xgb

from keras.models import Model
from keras.layers import Input, Dense, Embedding, Flatten, Concatenate, BatchNormalization, Dropout
from keras import Sequential, layers, regularizers, optimizers
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError




def baseline_model():
 pass
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

def train_xgb_model(params, X, y, X_val=None, y_val=None, eval_metric="rmse", early_stopping_rounds=5, verbose=True):
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
        num_boost_round=params.get("n_estimators", 5),
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose,
        evals_result=evals_result,  # Optionally track evaluation metrics during training
    )

    return model, evals_result


def initialize_keras_model(n_numeric_features: int) -> Model:
        """
        Initialize the Neural Network with random weights
        """

         # Categorical Inputs
        departement_input = Input(shape=(1,), name="departement_input")
        unique_city_id_input = Input(shape=(1,), name="unique_city_id_input")

        #  Embedding Layers
        departement_emb = Embedding(input_dim=91, output_dim=8, name="departement_embedding")(departement_input)
        unique_city_emb = Embedding(input_dim=33523, output_dim=16, name="unique_city_embedding")(unique_city_id_input)

        # Flatten embeddings
        departement_emb = Flatten()(departement_emb)
        unique_city_emb = Flatten()(unique_city_emb)

        #  Numeric Inputs
        numeric_input = Input(shape=(n_numeric_features,), name="numeric_input")

        # Concatenate embeddings with numeric inputs
        merged = Concatenate()([departement_emb, unique_city_emb, numeric_input])

        reg = regularizers.l1_l2(l2=0.005)

        x = Dense(100, activation="relu", kernel_regularizer=reg)(merged)
        x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(rate=0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Dropout(rate=0.1)(x)
        output = Dense(1, activation="linear")(x)

        model = Model(inputs=[departement_input, unique_city_id_input, numeric_input], outputs=output)


        print("✅ Model initialized with embeddings")

        return model


def compile_keras_model(model: Model, learning_rate=0.0001) -> Model:
        """
        Compile the Neural Network
        """

        def rmse(y_true, y_pred):
            diff = y_pred - y_true
            return tf.sqrt(tf.reduce_mean(tf.square(diff) + 1e-8))

        optimizer = optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0)
        model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=[rmse])

        print("✅ Model compiled")

        return model

def train_keras_model(
            model: Model,
            X: np.ndarray,
            y: np.ndarray,
            batch_size=256,
            patience=2,
            validation_data=None, # overrides validation_split
            validation_split=0.3
        ) -> Tuple[Model, dict]:
        """
        Fit the model and return a tuple (fitted_model, history)
        """
        print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

        es = EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )

        history = model.fit(
            X,
            y,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=100,
            batch_size=batch_size,
            callbacks=[es],
            verbose=1
        )

        print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_rmse']), 2)}")

        return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64,
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model" + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True,
    )

    loss = metrics["loss"]
    rmse = metrics["rmse"]

    print(f"✅ Model evaluated, RMSE: {round(rmse, 2)}")

    return metrics

