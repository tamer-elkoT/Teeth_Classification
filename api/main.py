from fastapi import FastAPI
from api.routers import router

app = FastAPI(title="Teeth Classification API")

app.include_router(router)

@app.get("/")
def health_check():
    return {"status": "The server is alive and running!"}