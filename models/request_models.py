from pydantic import BaseModel, HttpUrl
from typing import List

class HackrxRequest(BaseModel):
    documents: HttpUrl
    urls: List[str]
