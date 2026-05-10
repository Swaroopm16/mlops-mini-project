import os
import mlflow
import dagshub

dagshub_username = os.getenv("DAGSHUB_USERNAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if not dagshub_username or not dagshub_token:
    raise EnvironmentError(
        "DAGSHUB_USERNAME or DAGSHUB_TOKEN is missing"
    )

# Configure MLflow authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set tracking URI
mlflow.set_tracking_uri(
    "https://dagshub.com/Swaroopm16/mlops-mini-project.mlflow"
)