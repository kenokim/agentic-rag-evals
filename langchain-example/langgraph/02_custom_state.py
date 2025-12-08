from typing import List, TypedDict
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END

# 1. 커스텀 상태 정의
# 기본 MessagesState 대신 직접 TypedDict로 정의할 수도 있음
class AgentState(TypedDict):
    messages: List[BaseMessage]
    summary: str # 대화 요약본 저장

# 2. 노드 정의
def call_model(state: AgentState):
    # 실제로는 LLM 호출
    # 여기서는 모의 응답
    last_user_msg = state["messages"][-1].content
    response = f"Echo: {last_user_msg}"
    return {"messages": [AIMessage(content=response)]}

def summarize_conversation(state: AgentState):
    # 대화가 길어지면 요약하는 로직 (예시)
    msgs = state["messages"]
    summary = f"Total messages: {len(msgs)}"
    return {"summary": summary}

# 3. 그래프 구성
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("summarizer", summarize_conversation)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", "summarizer")
workflow.add_edge("summarizer", END)

# 4. 컴파일
app = workflow.compile()

# 5. 실행
try:
    inputs = {
        "messages": [HumanMessage(content="Hello LangGraph!")],
        "summary": ""
    }
    result = app.invoke(inputs)
    print("--- Execution Result ---")
    print(result)
except Exception as e:
    print(f"Error: {e}")

