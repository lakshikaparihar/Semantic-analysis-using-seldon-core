model_name = 'MovieModel'

import mlflow

# The default path where the MLflow autologging function stores the Keras model
run_id = 'e4347c93b22140f1b9888ad6f029d245'
artifact_path = "sklearn-model"
#model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=artifact_path)
model_uri = './mlruns/0/e4347c93b22140f1b9888ad6f029d245/artifacts/sklearn-model'
model_version = 1

model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

import time
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus

# Wait until the model is ready
def wait_until_ready(model_name, model_version):
  client = MlflowClient()
  for _ in range(10):
    model_version_details = client.get_model_version(
      name=model_name,
      version=model_version,
    )
    status = ModelVersionStatus.from_string(model_version_details.status)
    print("Model status: %s" % ModelVersionStatus.to_string(status))
    if status == ModelVersionStatus.READY:
      break
    time.sleep(1)

wait_until_ready(model_details.name, model_details.version)
