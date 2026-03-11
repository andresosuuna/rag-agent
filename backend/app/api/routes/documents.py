from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os
from app.services.pdf_service import process_pdf
from app.services.web_service import process_url
from pydantic import BaseModel

router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Receives a PDF file, processes it
    and saves embeddings to ChromaDB.
    """

    # Only accept PDF files
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )

    # Save file temporarily to disk for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = process_pdf(tmp_path)
        return {
            "status": "ok",
            "filename": file.filename,
            **result
        }
    finally:
        # Always delete the temp file even if an error occurs
        os.unlink(tmp_path)

class UrlRequest(BaseModel):
    url: str

@router.post("/url")
async def upload_url(request: UrlRequest):
    """
    Receives a URL, scrapes its content
    and saves embeddings to ChromaDB.
    """
    try:
        result = process_url(request.url)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not process URL: {str(e)}"
        )