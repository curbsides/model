from fastapi import FastAPI
from .routes.model_routes import router as model_router

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router, prefix="/model")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Healthcheck!"}
