import random
import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
from botocore.exceptions import NoCredentialsError, ClientError
import boto3

load_dotenv()

S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_REGION = os.getenv("S3_REGION")

# Define the router
router = APIRouter()

s3_client = boto3.client(
    "s3",
    region_name=S3_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
)

@router.get("/{filename}")
async def serve_image(filename: str):
    try:
        s3_object = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=f'images/{filename}')
        blob_data = s3_object["Body"].read()

        # run model here
        return {"result": False}


    except NoCredentialsError:
        print("ERROR")
        return {"result": False}
    except ClientError:
       print("ERROR")
       return {"result": False}