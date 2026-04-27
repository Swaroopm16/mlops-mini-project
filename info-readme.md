https://dagshub.com/Swaroopm16/mlops-mini-project.mlflow

import dagshub
dagshub.init(repo_owner='Swaroopm16', repo_name='mlops-mini-project', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)