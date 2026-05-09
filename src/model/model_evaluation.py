import os
import json
import pickle
import logging

import dagshub
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# =========================================================
# DagsHub + MLflow Configuration
# =========================================================

dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_username or not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN is missing"
    )

# MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# MLflow tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Swaroopm16/mlops-mini-project.mlflow"
)

# Initialize DagsHub
dagshub.init(
    repo_owner="Swaroopm16",
    repo_name="mlops-mini-project",
    mlflow=True,
)

# =========================================================
# Logging Configuration
# =========================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("model_evaluation_errors.log")
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# =========================================================
# Utility Functions
# =========================================================

def load_model(file_path: str):
    """Load trained model."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.debug("Model loaded from %s", file_path)
        return model

    except Exception as e:
        logger.error("Error loading model: %s", e)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data."""
    try:
        df = pd.read_csv(file_path)

        logger.debug("Data loaded from %s", file_path)
        return df

    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise


def evaluate_model(clf, X_test, y_test):
    """Evaluate model performance."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_pred_proba),
        }

        logger.debug("Evaluation metrics calculated")

        return metrics

    except Exception as e:
        logger.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str):
    """Save metrics JSON."""
    try:
        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.debug("Metrics saved to %s", file_path)

    except Exception as e:
        logger.error("Error saving metrics: %s", e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str):
    """Save experiment metadata."""
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path,
        }

        with open(file_path, "w") as file:
            json.dump(model_info, file, indent=4)

        logger.debug("Model info saved to %s", file_path)

    except Exception as e:
        logger.error("Error saving model info: %s", e)
        raise

# =========================================================
# Main Pipeline
# =========================================================

def main():

    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:

        try:

            # Load model and test data
            clf = load_model("./models/model.pkl")

            test_data = load_data(
                "./data/processed/test_bow.csv"
            )

            # Split features and labels
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test)

            # Save metrics locally
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters
            if hasattr(clf, "get_params"):

                params = clf.get_params()

                for param_name, param_value in params.items():
                    mlflow.log_param(
                        param_name,
                        param_value,
                    )

            # Log model artifact
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model",
            )

            # Save experiment info
            save_model_info(
                run.info.run_id,
                "model",
                "reports/experiment_info.json",
            )

            # Log artifacts
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact(
                "reports/experiment_info.json"
            )

            if os.path.exists(
                "model_evaluation_errors.log"
            ):
                mlflow.log_artifact(
                    "model_evaluation_errors.log"
                )

            logger.debug(
                "Model evaluation pipeline completed"
            )

        except Exception as e:

            logger.error(
                "Failed during model evaluation: %s",
                e,
            )

            raise


if __name__ == "__main__":
    main()