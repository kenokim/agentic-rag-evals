from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from service import get_vectorstore, load_parent_chunks, LLM_MODEL

# --- Tools ---

@tool
def search_child_chunks(query: str) -> List[dict]:
    """벡터 DB에서 자식 청크 검색. 가장 먼저 사용."""
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=5)
    return [
        {
            "content": doc.page_content,
            "parent_id": doc.metadata.get("parent_id"),
            "source": doc.metadata.get("source", "unknown")
        }
        for doc in results
    ]

@tool
def retrieve_parent_chunks(parent_ids: List[str]) -> List[str]:
    """parent_id로 전체 문맥(부모 청크) 조회."""
    docs = load_parent_chunks(parent_ids)
    return [doc.page_content for doc in docs]


class AgenticRAG:
    """
    Agent-based RAG pipeline using LangGraph.
    """
    def __init__(self):
        self.app = self._build_graph()
        
    def _build_graph(self):
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)
        tools = [search_child_chunks, retrieve_parent_chunks]
        llm_with_tools = llm.bind_tools(tools)

        def agent_node(state: MessagesState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        builder = StateGraph(MessagesState)
        builder.add_node("agent", agent_node)
        builder.add_node("tools", ToolNode(tools))

        builder.add_edge(START, "agent")
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
        
        return builder.compile()

    def get_answer(self, query: str) -> Dict[str, Any]:
        """
        에이전트 그래프를 실행하여 능동적 검색 및 답변 생성
        """
        system_prompt = """You are a helpful RAG assistant.
        1. First, ALWAYS search for relevant information using 'search_child_chunks'.
        2. Analyze the search results. If you need more context, use 'retrieve_parent_chunks'.
        3. Answer based ONLY on retrieved info.
        """

        inputs = {"messages": [SystemMessage(content=system_prompt), HumanMessage(content=query)]}
        final_state = self.app.invoke(inputs, config={"recursion_limit": 10})
        
        messages = final_state["messages"]
        answer = messages[-1].content
        
        # 출처 추출
        sources = []
        seen_sources = set()
        
        for msg in messages:
            if isinstance(msg, ToolMessage) and msg.name == "search_child_chunks":
                try:
                    import ast
                    results = ast.literal_eval(msg.content)
                    if isinstance(results, list):
                        for res in results:
                            if isinstance(res, dict) and "source" in res:
                                src_name = res["source"]
                                if src_name not in seen_sources:
                                    sources.append({
                                        "source": src_name,
                                        "page": 0,
                                        "content": res.get("content", "")[:100] + "..."
                                    })
                                    seen_sources.add(src_name)
                except:
                    pass
                    
        return {
            "answer": answer,
            "sources": sources
        }
