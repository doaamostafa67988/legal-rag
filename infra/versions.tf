terraform {
  required_version = ">= 1.6.0"

  required_providers {
    docker = {
      source  = "kreuzwerker/docker"
      version = "~> 3.0"
    }
  }
}

# The Docker daemon of this machine (the unix socket inside WSL). No cloud account is involved.
provider "docker" {}
