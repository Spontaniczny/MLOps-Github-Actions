import os
import boto3

from src.scripts.settings import Settings


def download_artifacts(settings: Settings):
    s3 = boto3.client('s3')

    BUCKET_NAME = settings.s3_bucket_name
    S3_OBJECT_KEY = settings.s3_model_dir
    LOCAL_DIR = settings.local_model_dir

    response = s3.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=S3_OBJECT_KEY
    )

    for obj in response.get("Contents", []):
        key = obj["Key"]

        if key.endswith("/"):
            continue

        local_path = os.path.join(
            LOCAL_DIR,
            os.path.relpath(key, S3_OBJECT_KEY)
        )

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(BUCKET_NAME, key, local_path)