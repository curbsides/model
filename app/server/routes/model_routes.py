import os
from fastapi import APIRouter
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
        return {"result": infer(blob_data)}


    except NoCredentialsError:
        print("ERROR")
        return {"result": False}
    except ClientError:
       print("ERROR")
       return {"result": False}


def infer(file_path):
    from transformers import ViTForImageClassification, AutoImageProcessor
    import torch
    from PIL import Image

    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", attn_implementation="sdpa", torch_dtype=torch.float16)
    model.to("cuda")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k",use_fast=True)
    inputs = image_processor(blob_data, return_tensors="pt")
    inputs.to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits
        
    predicted_label = logits.argmax(-1).item()
    returnbool predicted_label 
    print(model.config.id2label[predicted_label])