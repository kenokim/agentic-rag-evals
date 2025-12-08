import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException
from models import IngestResponse, ChatRequest, ChatResponse
import service
from simple_rag import SimpleRAG
from agentic_rag import AgenticRAG

router = APIRouter()

# 전역 인스턴스 초기화
simple_rag_system = SimpleRAG()
agentic_rag_system = AgenticRAG()

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Ingest ---

@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="PDF 문서 업로드 및 적재",
    description="PDF 파일을 업로드하여 텍스트를 추출하고, 청크로 분할한 뒤 벡터 데이터베이스와 로컬 파일 저장소에 적재합니다."
)
async def ingest_document(file: UploadFile = File(..., description="업로드할 PDF 파일")):
    """
    PDF 파일을 업로드하고 RAG 시스템에 적재합니다.
    """
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        chunks_count = service.ingest_document(file_path)
        
        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_count=chunks_count,
            message="문서가 성공적으로 적재되었습니다."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Chat ---

@router.post(
    "/chat/simple",
    response_model=ChatResponse,
    summary="단순 RAG 채팅 (Retrieve-Read)",
    description="전통적인 검색 기반 답변 생성 방식입니다. 질문과 유사한 문서를 검색하고, 이를 바탕으로 LLM이 답변을 생성합니다."
)
async def chat_simple(request: ChatRequest):
    """
    기본적인 검색 기반 답변 생성 (Simple RAG)
    """
    try:
        result = simple_rag_system.get_answer(request.query)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post(
    "/chat/agentic",
    response_model=ChatResponse,
    summary="에이전트 RAG 채팅 (Agentic)",
    description="LangGraph 기반의 에이전트가 작동합니다. 스스로 필요한 정보를 검색하고, 문맥을 파악하여 더 정교한 답변을 생성합니다."
)
async def chat_agentic(request: ChatRequest):
    """
    에이전트 기반의 능동적 검색 및 답변 생성 (Agentic RAG)
    """
    try:
        result = agentic_rag_system.get_answer(request.query)
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
