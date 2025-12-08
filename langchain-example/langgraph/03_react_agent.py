import logging
import os
from typing import TypedDict, Annotated, Sequence
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. 도구 정의
@tool
def search_web(query: str) -> str:
    """웹 검색을 수행합니다."""
    # 실제 검색 API 대신 모의 응답
    logger.info(f"Searching web for: {query}")
    return f"Result for {query}: Python 3.12 was released in late 2023."

@tool
def get_current_time() -> str:
    """현재 시간을 반환합니다."""
    return "2024-05-21 15:30:00"

tools = [search_web, get_current_time]

# 2. 모델 설정 (Gemini + Tools)
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
model_with_tools = model.bind_tools(tools)

# 3. 상태(State) 정의
class AgentState(TypedDict):
    # add_messages: 리스트에 메시지를 추가(append)하는 리듀서
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 4. 노드 정의
def reasoner(state: AgentState):
    """ReAct: 생각하고(Reason), 행동(Action)을 결정하는 노드"""
    logger.info("--- Reasoner Node ---")
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

# 5. 조건부 엣지 정의
def should_continue(state: AgentState):
    """도구 호출이 필요한지 판단"""
    last_message = state["messages"][-1]
    
    if last_message.tool_calls:
        logger.info("Decision: Call Tools")
        return "tools"
    
    logger.info("Decision: End")
    return END

# 6. 그래프 구성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent", reasoner)
workflow.add_node("tools", ToolNode(tools)) # LangGraph 내장 ToolNode 사용

# 엣지 연결
workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)
workflow.add_edge("tools", "agent") # 도구 실행 후 다시 agent로 돌아와 결과를 보고함

# 7. 컴파일
app = workflow.compile()

# 8. 실행
if __name__ == "__main__":
    try:
        print("--- ReAct Agent 실행 ---")
        query = "Python 최신 버전은 언제 나왔어?"
        inputs = {"messages": [HumanMessage(content=query)]}
        
        for event in app.stream(inputs):
            for key, value in event.items():
                print(f"\n[{key}]")
                last_msg = value["messages"][-1]
                if isinstance(last_msg, AIMessage):
                    print(f"AI: {last_msg.content}")
                    if last_msg.tool_calls:
                        print(f"Tools: {last_msg.tool_calls}")
                elif isinstance(last_msg, ToolMessage):
                    print(f"Tool Output: {last_msg.content}")

    except Exception as e:
        print(f"Error: {e}")

