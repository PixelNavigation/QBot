import httpx
import tempfile
import mimetypes
from typing import Optional

async def download_blob_to_temp_file(blob_url: str) -> Optional[str]:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(str(blob_url))
            response.raise_for_status()
            
            # Determine file extension
            content_type = response.headers.get('content-type', '')
            extension = mimetypes.guess_extension(content_type) or '.bin'
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
                tmp_file.write(response.content)
                return tmp_file.name
                
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None