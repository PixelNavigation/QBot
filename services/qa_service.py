import os
import openai
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import Dict, Any, List, Tuple
import tempfile
import shutil
from dotenv import load_dotenv
load_dotenv()

class DocumentQAService:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""]
        )
        self.vector_store = None
        self.temp_db_path = None

    def create_chunks_with_metadata(self, processed_document: Dict[str, Any]) -> List[Document]:
        documents = []
        for page in processed_document["pages"]:
            page_number = page["page_number"]
            page_text = page["full_text"]
            chunks = self.text_splitter.split_text(page_text)

            for chunk_idx, chunk in enumerate(chunks):
                chunk_lines = self._find_lines_for_chunk(chunk, page["lines"])
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "page_number": page_number,
                        "chunk_index": chunk_idx,
                        "line_numbers": [line["line_number"] for line in chunk_lines],
                        "chunk_length": len(chunk)
                    }
                )
                documents.append(doc)
        return documents

    def _find_lines_for_chunk(self, chunk_text: str, page_lines: List[Dict]) -> List[Dict]:
        chunk_lines = []
        chunk_words = set(chunk_text.lower().split())
        for line in page_lines:
            line_words = set(line["content"].lower().split())
            if len(chunk_words.intersection(line_words)) >= len(line_words) * 0.3:
                chunk_lines.append(line)
        return chunk_lines

    def setup_vector_store(self, processed_document: Dict[str, Any]):
        try:
            self.temp_db_path = tempfile.mkdtemp()
            documents = self.create_chunks_with_metadata(processed_document)
            client = chromadb.PersistentClient(path=self.temp_db_path)
            self.vector_store = Chroma(
                client=client,
                collection_name="document_chunks",
                embedding_function=self.embeddings,
            )
            self.vector_store.add_documents(documents)
            print(f"Created vector store with {len(documents)} chunks")
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            raise

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        return self.vector_store.similarity_search_with_score(query, k=k)

    def generate_answer(self, query: str, relevant_chunks: List[Document]) -> str:
        context = "\n\n".join(chunk.page_content for chunk in relevant_chunks)

        prompt = f"""You are a helpful assistant that answers questions based on the provided document content.

Context from document:
{context}

Question: {query}

Instructions:
- Answer the question using ONLY the information provided in the context above
- Be specific and accurate
- Quote directly from the context when possible
- If the information is not available in the context, say "The information is not available in the provided document"
- Keep your answer concise but complete

Answer:"""

        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on document content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return f"Error generating answer: {str(e)}"

    def cleanup(self):
        if self.temp_db_path and os.path.exists(self.temp_db_path):
            try:
                shutil.rmtree(self.temp_db_path)
                print("Cleaned up temporary vector store")
            except Exception as e:
                print(f"Error cleaning up vector store: {e}")

# Global instance
qa_service = DocumentQAService()

async def answer_query_with_rag(query: str, processed_document_data: Dict[str, Any]) -> str:
    try:
        if qa_service.vector_store is None:
            qa_service.setup_vector_store(processed_document_data)

        relevant_chunks_with_scores = qa_service.retrieve_relevant_chunks(query, k=5)
        relevant_chunks = [chunk for chunk, _ in relevant_chunks_with_scores]

        if not relevant_chunks:
            return "No relevant information found in the document."

        answer = qa_service.generate_answer(query, relevant_chunks)
        return answer

    except Exception as e:
        print(f"RAG error: {e}")
        return f"Error processing question: {str(e)}"

def cleanup_qa_service():
    qa_service.cleanup()
