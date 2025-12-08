from typing import TypedDict, Literal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

# 1. 상태(State) 정의
# messages 리스트를 상태로 가짐 (대화 기록)
from langgraph.graph import MessagesState

# 2. 도구(Tool) 정의
@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

tools = [multiply, add]

# 3. 모델 설정 (Tools 바인딩)
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# 4. 노드(Node) 함수 정의
def chatbot_node(state: MessagesState):
    """LLM을 호출하여 응답을 생성하는 노드"""
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 5. 조건부 엣지(Conditional Edge) 함수 정의
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """
    LLM의 마지막 메시지가 도구 호출을 포함하면 'tools'로,
    그렇지 않으면 종료('__end__')로 이동
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# 6. 그래프(Graph) 구성
builder = StateGraph(MessagesState)

# 노드 추가
builder.add_node("chatbot", chatbot_node)
builder.add_node("tools", ToolNode(tools)) # ToolNode는 미리 만들어진 노드 사용

# 엣지 연결
builder.add_edge(START, "chatbot") # 시작 -> 챗봇
builder.add_conditional_edges(
    "chatbot",
    should_continue,
)
builder.add_edge("tools", "chatbot") # 도구 실행 후 다시 챗봇으로 (결과 반영)

# 7. 그래프 컴파일
graph = builder.compile()

# 8. 실행 및 시각화 (옵션)
# 이미지로 저장: graph.get_graph().draw_mermaid_png()

try:
    print("--- LangGraph 실행 ---")
    initial_input = {"messages": [("user", "3 곱하기 4는 뭐야? 그리고 2를 더해줘.")]}
    
    # stream을 사용하여 중간 과정 확인
    for event in graph.stream(initial_input):
        for key, value in event.items():
            print(f"\n[{key}]")
            print(value["messages"][-1])
            
except Exception as e:
    print(f"Error: {e}")

