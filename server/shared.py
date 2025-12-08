import os
import json
import uuid
from typing import List
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

import pymupdf4llm

# --- Configuration ---
CHROMA_DB_DIR = "./chroma_db"
PARENT_STORE_DIR = "./parent_store"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"

# 디렉토리 생성
os.makedirs(PARENT_STORE_DIR, exist_ok=True)
os.makedirs(CHROMA_DB_DIR, exist_ok=True)

if not os.environ.get("GOOGLE_API_KEY"):
    print("경고: GOOGLE_API_KEY가 설정되지 않았습니다.")


# --- Shared Database Utilities ---

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

def get_vectorstore():
    """ChromaDB 벡터 저장소 인스턴스 반환"""
    return Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=get_embeddings(),
        collection_name="rag_collection"
    )

def save_parent_chunks(chunks: List[Document]):
    """부모 청크 로컬 저장"""
    for chunk in chunks:
        parent_id = chunk.metadata["parent_id"]
        safe_id = "".join([c for c in parent_id if c.isalnum() or c in ('-', '_')])
        file_path = os.path.join(PARENT_STORE_DIR, f"{safe_id}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }, f, ensure_ascii=False, indent=2)

def load_parent_chunks(parent_ids: List[str]) -> List[Document]:
    """부모 청크 로드"""
    documents = []
    for pid in parent_ids:
        safe_id = "".join([c for c in pid if c.isalnum() or c in ('-', '_')])
        file_path = os.path.join(PARENT_STORE_DIR, f"{safe_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                documents.append(Document(
                    page_content=data["page_content"],
                    metadata=data["metadata"]
                ))
    return documents

def ingest_document(file_path: str) -> int:
    """
    공통 문서 적재 로직: PDF -> Markdown -> Parent/Child Chunks -> Store
    """
    file_path_obj = Path(file_path)
    
    # 1. PDF -> Markdown
    md_text = pymupdf4llm.to_markdown(file_path)

    # 2. Parent Chunking (Header based)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    parent_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    parent_chunks = parent_splitter.split_text(md_text)
    
    if not parent_chunks:
        parent_chunks = [Document(page_content=md_text, metadata={})]

    # 3. Child Chunking
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    all_parent_chunks = []
    all_child_chunks = []

    for i, parent_chunk in enumerate(parent_chunks):
        unique_id = uuid.uuid4().hex[:8]
        parent_id = f"{file_path_obj.stem}_p{i}_{unique_id}"
        
        parent_chunk.metadata["parent_id"] = parent_id
        parent_chunk.metadata["source"] = file_path_obj.name
        all_parent_chunks.append(parent_chunk)
        
        child_chunks = child_splitter.split_documents([parent_chunk])
        for child in child_chunks:
            child.metadata["parent_id"] = parent_id
            child.metadata["source"] = file_path_obj.name
            all_child_chunks.append(child)

    if not all_child_chunks:
        return 0

    # 4. Store
    save_parent_chunks(all_parent_chunks)
    
    Chroma.from_documents(
        documents=all_child_chunks,
        embedding=get_embeddings(),
        persist_directory=CHROMA_DB_DIR,
        collection_name="rag_collection"
    )

    return len(all_child_chunks)
