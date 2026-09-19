import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "dl_lab-1"


def setup():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
