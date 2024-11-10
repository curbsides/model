from fastapi import FastAPI
from .routes.model_routes import router as model_router



app = FastAPI()

app.include_router(model_router, prefix="/model")

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Healthcheck!"}

from fastapi.middleware.cors import CORSMiddleware
from .routes.loc_routes import router as loc_router
from .routes.img_routes import router as img_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
