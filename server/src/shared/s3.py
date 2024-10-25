import json
import boto3
from botocore.config import Config
from .config import *

s3 = boto3.client(
    service_name="s3",
    aws_access_key_id=S3_ACCESS_KEY,
    aws_secret_access_key=S3_SECRET_KEY,
    endpoint_url=S3_ENDPOINT_URL,
    config=Config(s3={"addressing_style": "path"})
)

if not (S3_BUCKET_PICS in [bucket['Name'] for bucket in s3.list_buckets()["Buckets"]]):
    s3.create_bucket(Bucket=S3_BUCKET_PICS)

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": "*",
                "Action": ["s3:GetObject"],
                "Resource": [f"arn:aws:s3:::{S3_BUCKET_PICS}/*"]
            }
        ]
    }

    s3.put_bucket_policy(Bucket=S3_BUCKET_PICS, Policy=json.dumps(policy))