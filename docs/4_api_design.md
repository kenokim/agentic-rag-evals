# API Design Specification

RAG 서비스를 위한 FastAPI 엔드포인트 설계를 정의합니다.

## 1. Ingest API (문서 적재)
PDF 파일을 업로드받아 텍스트를 추출하고, 임베딩하여 Vector DB에 저장합니다.

### Endpoint Info
- **URL**: `/ingest`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`

### Request Parameters
| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `file` | File | 업로드할 PDF 파일 | Yes |

### Response Body (JSON)
```json
{
  "status": "success",
  "filename": "document.pdf",
  "chunks_count": 42,
  "message": "Successfully ingested document."
}
```

### 처리 로직 (Pipeline)
1.  업로드된 파일을 임시 디렉토리에 저장.
2.  `PyMuPDFLoader`를 사용하여 PDF 로딩.
3.  `RecursiveCharacterTextSplitter`로 텍스트 분할 (Chunking).
4.  `GoogleGenerativeAIEmbeddings` (models/embedding-001)로 임베딩 생성.
5.  `Chroma` Vector DB에 저장.

---

## 2. Chat API (질의응답)
사용자의 질문을 받아 RAG 파이프라인을 거쳐 답변을 생성합니다.

### Endpoint Info
- **URL**: `/chat`
- **Method**: `POST`
- **Content-Type**: `application/json`

### Request Body (JSON)
```json
{
  "query": "이 논문의 주요 주장은 무엇인가요?"
}
```

### Response Body (JSON)
```json
{
  "answer": "이 논문의 주요 주장은...",
  "sources": [
    {
      "source": "document.pdf",
      "page": 1,
      "content": "관련된 원본 텍스트 내용..."
    }
  ]
}
```

### 처리 로직 (Pipeline)
1.  사용자 Query 수신.
2.  Query를 임베딩하여 Vector DB(`Chroma`)에서 유사한 문서 청크 검색 (Retriever).
3.  검색된 문맥(Context)과 질문(Query)을 프롬프트 템플릿에 결합.
4.  LLM(Gemini Pro 권장)에 전송하여 답변 생성.
5.  답변과 함께 참조한 문서(Source) 정보를 반환.

---

## 3. Data Models (Pydantic)

```python
from pydantic import BaseModel
from typing import List

class ChatRequest(BaseModel):
    query: str

class SourceInfo(BaseModel):
    source: str
    page: int
    content: str

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
```

