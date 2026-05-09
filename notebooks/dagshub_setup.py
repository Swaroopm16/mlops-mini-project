import os
import dagshub
import mlflow

dagshub.init(
    repo_owner="Swaroopm16",
    repo_name="mlops-mini-project",
    mlflow=True
)

os.environ["MLFLOW_TRACKING_USERNAME"] = os.environ["DAGSHUB_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]

mlflow.set_tracking_uri(
    "https://dagshub.com/Swaroopm16/mlops-mini-project.mlflow"
)
