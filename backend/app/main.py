from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.documents import router as documents_router

app = FastAPI(
    title="RAG Agent API",
    description="Chatea con tus documentos",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "RAG Agent funcionando",
        "debug": settings.debug
    }