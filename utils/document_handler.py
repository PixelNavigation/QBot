import httpx
import os
import tempfile
import mimetypes
from typing import Optional

async def download_blob_to_temp_file(blob_url: str) -> Optional[str]: