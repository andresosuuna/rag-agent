from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.rag_chain import ask
import traceback

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    question: str
    collection_name: str = "documents"

@router.post("/")
async def chat(request: ChatRequest):
    """
    Receives a question and returns an answer
    with the sources used to generate it.
    """
    try:
        result = ask(
            question=request.question,
            collection_name=request.collection_name
        )
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )