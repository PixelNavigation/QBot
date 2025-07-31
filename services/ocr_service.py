import os
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from typing import Dict, Any
from starlette.concurrency import run_in_threadpool

async def process_document_with_ocr(file_path: str) -> Dict[str, Any]:
    endpoint = os.getenv("AZURE_DI_ENDPOINT")
    key = os.getenv("AZURE_DI_KEY")

    if not endpoint or not key:
        raise ValueError("Azure Document Intelligence credentials not configured")

    try:
        client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

        with open(file_path, "rb") as document:
            poller = await run_in_threadpool(client.begin_analyze_document,
                                             "prebuilt-layout",
                                             document=document,
                                             content_type="application/octet-stream")
            result = await run_in_threadpool(poller.result)

        pages = []
        full_text = ""

        if result.pages:
            for page_idx, page in enumerate(result.pages):
                page_text = ""
                words = []

                if page.lines:
                    for line in page.lines:
                        page_text += line.content + "\n"
                        if line.spans:
                            for span in line.spans:
                                words.append({
                                    "content": result.content[span.offset:span.offset + span.length],
                                    "offset": span.offset,
                                    "length": span.length
                                })

                pages.append({
                    "page_number": page_idx + 1,
                    "full_text": page_text.strip(),
                    "words": words
                })
                full_text += page_text + "\n"

        return {
            "full_document_text": full_text.strip(),
            "pages": pages,
            "total_pages": len(pages)
        }

    except Exception as e:
        print(f"OCR processing error: {e}")
        raise
