import os
import openai
from typing import Dict, Any

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

async def answer_query_with_rag(query: str, processed_document_data: Dict[str, Any]) -> Dict[str, Any]: