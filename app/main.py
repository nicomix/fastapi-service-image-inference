from typing import Union
from fastapi import FastAPI
from .routers import images

app = FastAPI()

app.include_router(
    router = images.router
)

@app.get("/")
def read_root():
    return {"Hello": "World"}