from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from service import get_vectorstore, LLM_MODEL

class SimpleRAG:
    """
    Standard Retrieve-Read RAG pipeline.
    """
    def __init__(self):
        self.vectorstore = get_vectorstore()
        self.llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
        
    def get_answer(self, query: str) -> Dict[str, Any]:
        """
        기본적인 검색 기반 답변 생성 (Retrieve-Read)
        """
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        prompt_template = """다음 문맥(Context)을 바탕으로 질문에 답변해 주세요.
        만약 문맥에서 답을 찾을 수 없다면 "제공된 문서에서 답변을 찾을 수 없습니다."라고 말해 주세요.

        Context:
        {context}

        Question:
        {input}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "input"]
        )
        
        combine_docs_chain = create_stuff_documents_chain(self.llm, PROMPT)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
        
        result = qa_chain.invoke({"input": query})
        
        # 출처 포맷팅
        sources = []
        for doc in result.get("context", []):
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": 0,
                "content": doc.page_content[:100] + "..."
            })
            
        return {
            "answer": result.get("answer", ""),
            "sources": sources
        }
