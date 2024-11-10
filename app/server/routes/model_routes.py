import random
from fastapi import APIRouter

# Define the router
router = APIRouter()

@router.get("/{filename}")
async def serve_image(filename: str):

    # load from S3

    # run on model

    outcome = random.random() < 0.30

    if outcome:
        return {"result": True}
    else:
        return {"result": False}
