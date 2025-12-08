import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 문서 로드 (웹페이지 예시)
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# 2. 문서 분할 (Splitting)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# 3. 벡터 저장소 생성 및 저장 (Indexing)
# GoogleGenerativeAIEmbeddings를 사용하여 벡터화 (text-embedding-004)
vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/text-embedding-004"))

# 4. 검색기(Retriever) 생성
retriever = vectorstore.as_retriever()

# 5. RAG 프롬프트 로드 (LangSmith Hub 사용 가능)
# 직접 정의:
from langchain_core.prompts import ChatPromptTemplate
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 6. LLM 설정
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# 7. 문서 포맷팅 헬퍼 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 8. RAG 체인 구성
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 9. 실행
try:
    response = rag_chain.invoke("What is Task Decomposition?")
    print("--- RAG 답변 ---")
    print(response)
except Exception as e:
    print(f"Error: {e}")

