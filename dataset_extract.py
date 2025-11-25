import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

# Settings
bucket = "aft-vbi-pds"
img_prefix = "bin-images/"
meta_prefix = "metadata/"
target_count = 10000

# Create folders
os.makedirs("bin-images", exist_ok=True)
os.makedirs("metadata", exist_ok=True)

# Connect to S3
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

count = 0
continuation_token = None

while count < target_count:
    if continuation_token:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=img_prefix, MaxKeys=1000, ContinuationToken=continuation_token
        )
    else:
        response = s3.list_objects_v2(
            Bucket=bucket, Prefix=img_prefix, MaxKeys=1000
        )

    for obj in response.get("Contents", []):
        key = obj["Key"]
        filename = os.path.basename(key)

        if not filename.endswith(".jpg"):
            continue

        img_path = f"bin-images/{filename}"
        json_path = f"metadata/{filename.replace('.jpg', '.json')}"

        if not os.path.exists(img_path):
            try:
                s3.download_file(bucket, key, img_path)
            except Exception as e:
                print(f"IMG DL Error {filename}: {e}")
                continue

        json_key = meta_prefix + filename.replace(".jpg", ".json")
        if not os.path.exists(json_path):
            try:
                s3.download_file(bucket, json_key, json_path)
            except Exception as e:
                print(f"JSON DL Error {filename}: {e}")
                continue

        count += 1
        print(f"[{count}/{target_count}] {filename}")

        if count >= target_count:
            break

    continuation_token = response.get("NextContinuationToken")
    if not continuation_token:
        break

print(" Done!")
