"""Create the artifact bucket if it does not exist yet (works with any S3-compatible store)."""

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"], region_name="us-east-1")

for _ in range(60):
    try:
        s3.head_bucket(Bucket=BUCKET)
        print(f"bucket {BUCKET} exists", flush=True)
        break
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchBucket"):
            s3.create_bucket(Bucket=BUCKET)
            print(f"bucket {BUCKET} created", flush=True)
            break
        print(f"object store answered {code}, retrying", flush=True)
    except Exception as exc:  # the store is not reachable yet
        print(f"waiting for the object store ({type(exc).__name__})", flush=True)
    time.sleep(1)
else:
    sys.exit("object store never became reachable")
