output "mlflow_url" {
  description = "MLflow UI and tracking URI."
  value       = "http://localhost:${var.mlflow_port}"
}

output "s3_endpoint" {
  description = "S3 API of the object store (MLflow artifacts and the DVC remote)."
  value       = "http://localhost:${var.s3_port}"
}

output "console_url" {
  description = "Object-store web console."
  value       = "http://localhost:${var.console_port}"
}

output "dvc_remote" {
  description = "Commands that point DVC at the bucket this stack created (keys go in --local config)."
  value       = "dvc remote add -d storage s3://${var.dvc_bucket} && dvc remote modify storage endpointurl http://localhost:${var.s3_port}"
}

output "containers" {
  value = [
    docker_container.postgres.name,
    docker_container.s3.name,
    docker_container.mlflow.name,
  ]
}
