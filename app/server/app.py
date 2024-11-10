from fastapi import FastAPI
from .routes.model_routes import router as model_router

app = FastAPI()

app.include_router(model_router, prefix="/model")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Healthcheck!"}
