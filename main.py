import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.request_models import HackrxRequest
from models.response_models import HackexResponse
from utils.document_handler import download_blob_to_temp_file
from services.ocr_service import process_document_with_ocr
from services.qa_service import answer_query_with_rag

load_dotenv()

app = FastAPI(
    title="Hackex API",
    description="API for Hackex, a document processing and question-answering service.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def authenticate(auth_header: str):
    expected_key = os.getenv("HACKRX_API_KEY")
    if not expected_key:
        raise HTTPException(status_code=500, detail="API key not configured.")

    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    
    token = auth_header.replace("Bearer ", "")
    if token != expected_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
@app.post("/hackrx/run", response_model=HackexResponse)
async def run_hackrx(
    request: HackrxRequest,
    authorization: str = Header(None)
):
    authenticate(authorization)

    temp_file_path=None
    try:
        temp_file_path = download_blob_to_temp_file(request.document_url)
        if not temp_file_path:
            raise HTTPException(status_code=400, detail="Failed to download document")

        processed_document = process_document_with_ocr(temp_file_path)
        if not processed_document:
            raise HTTPException(status_code=400, detail="OCR processing failed")

        answers = []
        for idx, query in enumerate(request.queries):
            answer = await answer_query_with_rag(query, processed_document)
            if not answer:
                raise HTTPException(status_code=400, detail=f"Failed to answer query {idx + 1}")
            answers.append(answer['answer'])

        return HackexResponse(answers=answers)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
