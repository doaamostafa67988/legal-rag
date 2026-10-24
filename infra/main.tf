# The same three services as docker/docker-compose.tracking.yml, declared as code:
# PostgreSQL (run metadata) + RustFS (S3-compatible artifact store) + the MLflow server.

resource "docker_network" "tracking" {
  name = "${var.name_prefix}-net"
}

resource "docker_volume" "pgdata" {
  name = "${var.name_prefix}-pgdata"
}

resource "docker_volume" "s3data" {
  name = "${var.name_prefix}-s3data"
}

# keep_locally: `terraform destroy` removes containers and volumes, not the (shared) images.
resource "docker_image" "postgres" {
  name         = var.postgres_image
  keep_locally = true
}

resource "docker_image" "rustfs" {
  name         = var.rustfs_image
  keep_locally = true
}

resource "docker_image" "mlflow" {
  name         = "${var.name_prefix}-mlflow:${var.mlflow_version}"
  keep_locally = true

  build {
    context    = "${path.module}/../docker/mlflow"
    build_args = { MLFLOW_VERSION = var.mlflow_version }
  }

  # rebuild when the image recipe changes
  triggers = {
    dockerfile    = filesha1("${path.module}/../docker/mlflow/Dockerfile")
    ensure_bucket = filesha1("${path.module}/../docker/mlflow/ensure_bucket.py")
  }
}

resource "docker_container" "postgres" {
  name    = "${var.name_prefix}-postgres"
  image   = docker_image.postgres.image_id
  restart = "unless-stopped"
  wait    = true

  env = [
    "POSTGRES_USER=${var.db_user}",
    "POSTGRES_PASSWORD=${var.db_password}",
    "POSTGRES_DB=${var.db_name}",
  ]

  networks_advanced {
    name    = docker_network.tracking.name
    aliases = ["postgres"]
  }

  volumes {
    volume_name    = docker_volume.pgdata.name
    container_path = "/var/lib/postgresql/data"
  }

  healthcheck {
    test     = ["CMD-SHELL", "pg_isready -U ${var.db_user} -d ${var.db_name}"]
    interval = "5s"
    timeout  = "3s"
    retries  = 10
  }
}

resource "docker_container" "s3" {
  name    = "${var.name_prefix}-s3"
  image   = docker_image.rustfs.image_id
  restart = "unless-stopped"

  env = [
    "RUSTFS_ACCESS_KEY=${var.s3_access_key}",
    "RUSTFS_SECRET_KEY=${var.s3_secret_key}",
    "RUSTFS_VOLUMES=/data",
    "RUSTFS_CONSOLE_ENABLE=true",
  ]

  ports {
    internal = 9000
    external = var.s3_port
  }

  ports {
    internal = 9001
    external = var.console_port
  }

  networks_advanced {
    name    = docker_network.tracking.name
    aliases = ["s3"] # MLflow and the bucket job reach the store as http://s3:9000
  }

  volumes {
    volume_name    = docker_volume.s3data.name
    container_path = "/data"
  }
}

resource "docker_container" "mlflow" {
  name    = "${var.name_prefix}-mlflow"
  image   = docker_image.mlflow.image_id
  restart = "unless-stopped"
  wait    = true

  # ensure_bucket.py creates the artifact bucket (retrying until the store is up), then the server starts
  command = [
    "sh", "-c",
    "python /ensure_bucket.py && exec mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri postgresql+psycopg2://${var.db_user}:${var.db_password}@postgres:5432/${var.db_name} --artifacts-destination s3://${var.artifact_bucket} --serve-artifacts",
  ]

  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://s3:9000",
    "MLFLOW_ARTIFACT_BUCKET=${var.artifact_bucket}",
    "AWS_ACCESS_KEY_ID=${var.s3_access_key}",
    "AWS_SECRET_ACCESS_KEY=${var.s3_secret_key}",
    "MLFLOW_SERVER_ALLOWED_HOSTS=${var.mlflow_allowed_hosts}",
    "MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false",
  ]

  ports {
    internal = 5000
    external = var.mlflow_port
  }

  networks_advanced {
    name    = docker_network.tracking.name
    aliases = ["mlflow"]
  }

  healthcheck {
    test         = ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')"]
    interval     = "5s"
    timeout      = "3s"
    retries      = 20
    start_period = "20s"
  }

  depends_on = [docker_container.postgres, docker_container.s3]
}

# One-shot job: creates the bucket the DVC remote uses, with the same script MLflow uses for its own.
resource "docker_container" "dvc_bucket" {
  name     = "${var.name_prefix}-dvc-bucket"
  image    = docker_image.mlflow.image_id
  command  = ["python", "/ensure_bucket.py"]
  restart  = "no"
  must_run = false # it exits when the bucket exists; Terraform treats that as success

  env = [
    "MLFLOW_S3_ENDPOINT_URL=http://s3:9000",
    "MLFLOW_ARTIFACT_BUCKET=${var.dvc_bucket}",
    "AWS_ACCESS_KEY_ID=${var.s3_access_key}",
    "AWS_SECRET_ACCESS_KEY=${var.s3_secret_key}",
  ]

  networks_advanced {
    name = docker_network.tracking.name
  }

  depends_on = [docker_container.s3]
}
