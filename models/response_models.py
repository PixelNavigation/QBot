from pydantic import BaseModel
from typing import List

class HackexResponse(BaseModel):
    answers: List[str]