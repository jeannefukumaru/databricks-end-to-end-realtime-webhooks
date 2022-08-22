import os
from mlflow.tracking import MlflowClient
import mlflow
import shutil

MLFLOW_TRACKING_URI = "databricks"
MODEL_NAME = "taxi_fare_regressor"
MODEL_VERSION = 1

client = MlflowClient(MLFLOW_TRACKING_URI)
model_version = client.get_model_version(
  name = MODEL_NAME,
  version = MODEL_VERSION
)
run_id = model_version.run_id
print(f"run id: {run_id}")

# Download the artifact to local storage.
local_dir = "ml_artifacts"
if not os.path.isdir(local_dir):
    os.mkdir(local_dir)
remote_dir = "train/model"
print(f"local_dir: {local_dir}")
print(f"local dir full path: {os.path.abspath(local_dir)}")
local_path = client.download_artifacts(run_id, remote_dir, local_dir)
full_path = f"{local_dir}/{remote_dir}"
print(f"Artifacts downloaded in {full_path}: {os.listdir(full_path)}")

print(f"copying downloaded model artifacts to docker folder")
dest_path = "./service/ml_artifacts/train/model"
if not os.path.isdir(dest_path):
  os.makedirs(dest_path)
for file in os.listdir(full_path):
  if file != "code":
    shutil.copy2(os.path.join(full_path, file), dest_path)            
print(f"artifacts moved to service folder {dest_path} {os.listdir(dest_path)}")