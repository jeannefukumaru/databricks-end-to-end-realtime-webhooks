terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "0.5.1"
    }
  }
}

provider "databricks" {
  host  = var.workspace_url
  token = var.pat_token
}

resource "databricks_mlflow_webhook" "url" {
  events      = ["MODEL_VERSION_TRANSITIONED_STAGE"]
  model_name  = "taxi_fare_regressor"
  description = "URL webhook trigger"
  http_url_spec {
    url = format("https://%s/api/HttpTrigger?", var.trigger_url)
  }
}