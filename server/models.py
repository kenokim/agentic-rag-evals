from pydantic import BaseModel, Field
from typing import List

class ChatRequest(BaseModel):
    query: str = Field(..., description="사용자의 질문", example="이 문서의 주요 내용이 뭐야?")

class SourceInfo(BaseModel):
    source: str = Field(..., description="문서 파일명")
    page: int = Field(..., description="페이지 번호 (PDF 등)")
    content: str = Field(..., description="참고한 문서의 내용 일부")

class ChatResponse(BaseModel):
    answer: str = Field(..., description="LLM이 생성한 답변")
    sources: List[SourceInfo] = Field(..., description="답변 생성에 사용된 출처 목록")

class IngestResponse(BaseModel):
    status: str = Field(..., description="처리 상태 (success/error)")
    filename: str = Field(..., description="처리된 파일명")
    chunks_count: int = Field(..., description="생성된 청크(Chunk) 개수")
    message: str = Field(..., description="처리 결과 메시지")
