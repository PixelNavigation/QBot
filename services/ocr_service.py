import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any

async def process_document_with_ocr(file_path: str) -> Dict[str, Any]: