# LangChain RAG Ingestion Pipeline (PDF Example)

LangChain을 사용하여 PDF 논문과 같은 문서를 Vector DB에 적재(Ingest)하는 파이프라인은 일반적으로 **Load -> Split -> Embed -> Store**의 4단계로 구성됩니다.

## 전체 파이프라인 개요

1.  **Load (문서 로딩)**: PDF 파일을 읽어와 텍스트 데이터로 변환합니다.
2.  **Split (청킹/분할)**: 긴 텍스트를 LLM의 컨텍스트 윈도우에 맞게 작은 조각(Chunk)으로 나눕니다.
3.  **Embed (임베딩)**: 텍스트 조각을 숫자 벡터로 변환합니다.
4.  **Store (저장)**: 벡터와 원본 텍스트를 Vector DB에 저장합니다.

---

## 단계별 상세 구현 (Python 예시)

이 예시는 앞서 추천한 **PyMuPDFLoader**와 **ChromaDB**를 사용하는 구성을 가정합니다.

### 사전 요구사항
```bash
pip install langchain langchain-community langchain-google-genai chromadb pymupdf
```

### 1. Document Loading (문서 로딩)
PDF 처리에 속도가 빠르고 메타데이터 추출이 용이한 `PyMuPDFLoader`를 사용합니다.

```python
from langchain_community.document_loaders import PyMuPDFLoader

# PDF 파일 경로 지정
file_path = "./data/paper.pdf"

# 로더 초기화 및 문서 로드
loader = PyMuPDFLoader(file_path)
documents = loader.load()

print(f"로드된 페이지 수: {len(documents)}")
print(documents[0].page_content[:100]) # 첫 페이지 내용 일부 확인
```

### 2. Text Splitting (텍스트 분할)
문맥을 최대한 유지하면서 자르기 위해 `RecursiveCharacterTextSplitter`를 주로 사용합니다.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 청크 하나당 최대 문자 수
    chunk_overlap=200,    # 청크 간 중복되는 문자 수 (문맥 단절 방지)
    separators=["\n\n", "\n", " ", ""] # 자르는 기준 우선순위
)

splits = text_splitter.split_documents(documents)

print(f"분할된 청크 수: {len(splits)}")
```

### 3. Embedding & 4. Vector Store (임베딩 및 저장)
Google Gemini 임베딩 모델을 정의하고, 이를 통해 텍스트를 벡터화하여 ChromaDB에 저장합니다.

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# 임베딩 모델 설정 (API Key 필요)
# GOOGLE_API_KEY 환경변수가 설정되어 있어야 합니다.
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Vector DB 생성 및 데이터 저장
# persist_directory를 지정하면 로컬에 DB 파일이 저장되어 재사용 가능
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embedding_model,
    collection_name="paper_collection",
    persist_directory="./chroma_db"
)

print("Vector DB 저장 완료!")
```

---

## 전체 통합 코드 예시

```python
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

def ingest_pdf(pdf_path: str):
    # 1. Load
    print("Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    
    # 2. Split
    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # 3. Embed & 4. Store
    print("Embedding and Storing...")
    # 모델명 명시: models/embedding-001
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print(f"Done! {len(splits)} chunks stored.")
    return vectorstore

if __name__ == "__main__":
    # API 키 설정 (실제 환경에서는 .env 파일 사용 권장)
    # os.environ["GOOGLE_API_KEY"] = "AIza..."
    
    ingest_pdf("data/my_paper.pdf")
```

### 추가 고려사항

*   **메타데이터 활용**: `PyMuPDFLoader`는 페이지 번호, 파일 경로 등의 메타데이터를 자동으로 포함합니다. 검색 시 이를 활용해 필터링(예: 특정 페이지만 검색)할 수 있습니다.
*   **한글 처리**: 한글 문서는 `chunk_size`를 영어보다 조금 작게 잡거나(약 500~800), `KoNLPy` 등을 활용한 커스텀 스플리터를 고려할 수도 있지만, 최신 `RecursiveCharacterTextSplitter`로도 대부분 충분합니다.

