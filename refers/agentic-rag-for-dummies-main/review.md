# Agentic RAG for Dummies 기술 스택 및 기능 분석

이 프로젝트는 PDF 문서를 처리하여 에이전트 기반 검색 증강 생성(RAG) 시스템을 구축하는 과정을 다루며, 주로 **LangChain 생태계**를 적극적으로 활용하고 있습니다.

## 1. 문서 전처리 (PDF Processing)
PDF 파일을 LLM이 이해하기 쉬운 형식으로 변환하는 단계입니다. LangChain이 아닌 전용 PDF 라이브러리를 사용합니다.
- **라이브러리**: `pymupdf`, `pymupdf4llm`
- **기능**:
    - PDF 문서를 레이아웃과 헤더 정보를 유지하며 **Markdown** 형식으로 변환
    - 이미지 제외 및 텍스트 정제

## 2. 문서 청킹 (Chunking)
변환된 문서를 검색과 답변 생성에 최적화된 크기로 나누는 단계입니다. **부모-자식(Parent-Child) 청킹 전략**을 사용합니다.
- **라이브러리**: `langchain_text_splitters`
- **활용 모듈**:
    - `MarkdownHeaderTextSplitter`: 마크다운 헤더(`#`, `##` 등)를 기준으로 큰 문맥(부모 청크)을 분리
    - `RecursiveCharacterTextSplitter`: 부모 청크를 다시 검색에 적합한 작은 크기(자식 청크)로 분할
- **특징**: 자식 청크로 정밀하게 검색하고, 답변 생성 시에는 원본 맥락이 담긴 부모 청크를 참조합니다.

## 3. 임베딩 (Embedding)
텍스트를 벡터로 변환하여 의미 기반 검색이 가능하게 합니다. **하이브리드 검색(Dense + Sparse)**을 지원합니다.
- **라이브러리**: `langchain_huggingface`, `langchain_qdrant`
- **활용 모듈**:
    - **Dense Embedding**: `HuggingFaceEmbeddings` (문맥적 의미 파악)
    - **Sparse Embedding**: `FastEmbedSparse` (키워드 매칭 보완)

## 4. 벡터 저장소 (Vector Store)
임베딩된 데이터를 저장하고 검색하는 데이터베이스입니다.
- **라이브러리**: `langchain_qdrant`, `qdrant_client`
- **활용 모듈**:
    - `QdrantVectorStore`: LangChain 인터페이스를 통해 Qdrant DB와 연동
    - **저장 구조**: 
        - 자식 청크(벡터) -> Qdrant DB 저장
        - 부모 청크(원본 텍스트) -> 별도 파일 스토어(JSON) 저장

## 5. 검색 및 도구 (Retrieval & Tools)
에이전트가 실제로 검색을 수행하고 데이터를 가져오는 방식입니다. **검색 로직 자체도 LangChain을 기반으로 작동**합니다.
- **라이브러리**: `langchain_core`, `langchain_qdrant`
- **활용 모듈**:
    - `similarity_search`: LangChain의 `VectorStore` 인터페이스를 통해 벡터 DB 검색 실행
    - `@tool`: 검색 함수(`search_child_chunks`, `retrieve_parent_chunks`)를 에이전트용 도구로 변환하는 데코레이터
- **특징**: LLM이 판단하여 필요 시 이 도구들을 호출하는 **능동적 검색(Active Retrieval)** 방식입니다.

## 6. LLM 및 에이전트 (LLM & Agent)
사용자의 질문을 이해하고 검색된 문서를 바탕으로 답변을 생성하며, 흐름을 제어합니다.
- **라이브러리**: `langchain_ollama`, `langgraph`
- **활용 모듈**:
    - `ChatOllama`: 로컬 LLM(Qwen 등) 구동 및 LangChain 연동
    - `LangGraph`: 에이전트의 워크플로우(검색 -> 판단 -> 답변) 제어 및 상태 관리
