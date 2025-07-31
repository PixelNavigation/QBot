import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from models.request_models import HackrxRequest
from models.response_models import HackexResponse
from utils.document_handler import download_blob_to_tempfile
from services.ocr_service import process_document_with_ocr
from services.qa_service import answer_query_with_rag