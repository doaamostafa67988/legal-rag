variable "name_prefix" {
  description = "Prefix of every resource this stack creates, so it never collides with the Compose stack."
  type        = string
  default     = "legal-rag-tf"
}

variable "mlflow_version" {
  description = "MLflow version baked into the server image (same as docker/docker-compose.tracking.yml)."
  type        = string
  default     = "3.17.0"
}

variable "postgres_image" {
  type    = string
  default = "postgres:16-alpine"
}

variable "rustfs_image" {
  description = "S3-compatible object store. Pin it to a digest once it works (docker images --digests)."
  type        = string
  default     = "rustfs/rustfs:latest"
}

# Host ports differ from the Compose stack (5000 / 9000 / 9001) so both can run at the same time and
# `terraform destroy` never touches the real experiment history. To make this stack replace the
# Compose one, stop that first and set the three ports to 5000 / 9000 / 9001 (terraform.tfvars).
variable "mlflow_port" {
  type    = number
  default = 5001
}

variable "s3_port" {
  type    = number
  default = 9100
}

variable "console_port" {
  type    = number
  default = 9101
}

variable "db_name" {
  type    = string
  default = "mlflow"
}

variable "db_user" {
  type    = string
  default = "mlflow"
}

# Local-only defaults, like the Compose file. Override them in terraform.tfvars (git-ignored).
variable "db_password" {
  type      = string
  default   = "mlflow"
  sensitive = true
}

variable "s3_access_key" {
  type      = string
  default   = "rustfsadmin"
  sensitive = true
}

variable "s3_secret_key" {
  type      = string
  default   = "rustfsadmin"
  sensitive = true
}

variable "artifact_bucket" {
  description = "Bucket MLflow stores artifacts in."
  type        = string
  default     = "mlflow-artifacts"
}

variable "dvc_bucket" {
  description = "Bucket used as a DVC remote."
  type        = string
  default     = "dvc"
}

variable "mlflow_allowed_hosts" {
  description = "Host headers the MLflow server accepts (DNS-rebinding protection in MLflow 3.5+). Never *."
  type        = string
  default     = "localhost:*,127.0.0.1:*,mlflow:*"
}
