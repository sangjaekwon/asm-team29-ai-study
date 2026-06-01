from fastapi import FastAPI

from api.agent3 import router as agent3_router

app = FastAPI(
        title="Agent3 API",
        version="1.0.0"
        )

app.include_router(
        agent3_router,
        prefix="/agent3",
        tags=["Agent3"]
        )

@app.get("/")
def root():
    return {"message": "HI"}

