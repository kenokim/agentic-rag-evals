import os
import bs4
from typing import List

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.retrievers import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 로깅
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 설정 ---
# 예제를 위해 웹 문서를 사용하지만, PDF 로더로 교체 가능
URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CHROMA_PATH = "./chroma_db_advanced"

def main():
    # 1. 문서 로드 및 분할
    logger.info("Loading & Splitting...")
    loader = WebBaseLoader(
        web_paths=(URL,),
        bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header")))
    )
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)

    # 2. 임베딩 및 저장
    embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding) # 메모리 모드

    # 3. 고급 검색기 (Multi-Query Retriever)
    # 사용자의 질문을 다양한 관점에서 3개의 다른 질문으로 변환하여 검색
    # -> 단일 쿼리 검색의 한계를 극복 (Query Expansion)
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    
    logger.info("Setting up Multi-Query Retriever...")
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(),
        llm=llm
    )

    # 4. RAG 체인
    template = """Answer the question based ONLY on the following context:
    {context}
    
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    
    chain = (
        {"context": retriever_from_llm, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. 실행
    query = "What are the limitations of current LLM agents?"
    print(f"\n[질문]: {query}")
    print("\n--- 답변 생성 중 (Multi-Query 검색) ---")
    
    # 로깅 레벨을 조정하여 내부적으로 생성된 쿼리를 볼 수 있음
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)
    
    response = chain.invoke(query)
    print(f"\n[답변]:\n{response}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")

