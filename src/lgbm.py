from pathlib import Path

import lightgbm as lgb
import mlflow

from src import tracking
from src.data import feature_columns
from src.metrics import evaluate

PARAMS = {
    "objective": "mae",
    "metric": ["mae", "rmse"],  
    "n_estimators": 5000, 
    "learning_rate": 0.3,
    "num_leaves": 1000,
    "max_depth": -1,
    "max_bin": 511,
    "min_child_samples": 100,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS = 100
MODELS_DIR = Path("models")


def train(train_part, val_part, params=PARAMS, run_name="lgbm"):
    features = feature_columns(train_part)
    X_train, y_train = train_part[features], train_part["pressure"]
    X_val, y_val = val_part[features], val_part["pressure"]
    val_mask = (val_part["u_out"] == 0).astype(int)

    tracking.setup()
    mlflow.lightgbm.autolog()

    with mlflow.start_run(run_name=run_name, log_system_metrics=True):
        mlflow.log_params(
            {
                "n_features": len(features),
                "train_rows": len(train_part),
                "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            }
        )
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train,
            y_train,
            eval_X=(X_train, X_val),
            eval_y=(y_train, y_val),
            eval_names=["train", "val"],
            eval_sample_weight=[None, val_mask],
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True),
                lgb.log_evaluation(250),
            ],
        )
        MODELS_DIR.mkdir(exist_ok=True)
        model.booster_.save_model(MODELS_DIR / f"{run_name}.txt")

        metrics = evaluate(y_val, model.predict(X_val), val_part["u_out"])
        mlflow.log_metrics(metrics)

    return model, metrics
