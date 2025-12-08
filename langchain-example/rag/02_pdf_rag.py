import os
from typing import List
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 설정 ---
# 1. 실제 사용할 PDF 경로 (예시용 더미 경로)
PDF_PATH = "sample_book.pdf" 
CHROMA_PATH = "./chroma_db"

def main():
    # 0. PDF 파일 존재 확인 (데모용)
    if not os.path.exists(PDF_PATH):
        logger.warning(f"PDF 파일이 없습니다: {PDF_PATH}")
        logger.warning("이 예제를 실행하려면 'sample_book.pdf' 파일을 같은 디렉토리에 두세요.")
        return

    # 1. 문서 로드 (Loading)
    logger.info("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    logger.info(f"Loaded {len(docs)} pages.")

    # 2. 텍스트 분할 (Splitting)
    # 두꺼운 책은 문맥 유지를 위해 chunk_overlap을 적절히 주는 것이 중요
    logger.info("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks.")

    # 3. 임베딩 및 인덱싱 (Indexing)
    logger.info("Indexing to Vector Store...")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Chroma DB에 저장 (Disk에 영구 저장)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PATH
    )
    logger.info("Indexing completed.")

    # 4. 검색기 생성 (Retrieval)
    # k=5: 상위 5개 관련 청크 검색
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 5. RAG 체인 구성 (Generation)
    template = """
    당신은 친절한 책 도우미 봇입니다. 아래 [문맥]을 바탕으로 질문에 답변해주세요.
    문맥에 없는 내용은 "책 내용에서 찾을 수 없습니다"라고 말해주세요.

    [문맥]
    {context}

    [질문]
    {question}

    [답변]
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 6. 질의응답 루프
    print("\n=== 책 RAG 챗봇 (종료하려면 'q' 입력) ===")
    while True:
        query = input("\n질문: ")
        if query.lower() in ["q", "quit", "exit"]:
            break
        
        try:
            response = rag_chain.invoke(query)
            print(f"\n답변: {response}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()

