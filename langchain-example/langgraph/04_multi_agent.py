import operator
from typing import Annotated, Sequence, TypedDict, List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END

# --- 1. 모델 설정 ---
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- 2. 상태 정의 ---
class ResearchState(TypedDict):
    task: str # 사용자 요청 작업
    initial_research: str # 1차 조사 결과
    critique: str # 비평 내용
    final_result: str # 최종 결과
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- 3. 에이전트(노드) 정의 ---

# A. Researcher Agent: 정보 수집 및 초안 작성
def researcher(state: ResearchState):
    print("--- Researcher: 조사 및 초안 작성 중 ---")
    task = state["task"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 꼼꼼한 연구원입니다. 주어진 주제에 대해 조사하고 짧은 요약 보고서를 작성하세요."),
        ("human", "{task}")
    ])
    chain = prompt | model
    response = chain.invoke({"task": task})
    
    return {"initial_research": response.content, "messages": [response]}

# B. Reviewer Agent: 비평 및 피드백
def reviewer(state: ResearchState):
    print("--- Reviewer: 검토 중 ---")
    research = state["initial_research"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 엄격한 편집자입니다. 연구원의 보고서를 읽고 부족한 점이나 개선할 점을 지적하세요."),
        ("human", "다음 보고서를 검토해줘:\n\n{research}")
    ])
    chain = prompt | model
    response = chain.invoke({"research": research})
    
    return {"critique": response.content, "messages": [response]}

# C. Writer Agent: 최종 수정
def writer(state: ResearchState):
    print("--- Writer: 최종 수정 중 ---")
    task = state["task"]
    research = state["initial_research"]
    critique = state["critique"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 전문 작가입니다. 연구원의 초안과 편집자의 피드백을 종합하여 최종 보고서를 작성하세요."),
        ("human", "주제: {task}\n\n초안:\n{research}\n\n피드백:\n{critique}")
    ])
    chain = prompt | model
    response = chain.invoke({"task": task, "research": research, "critique": critique})
    
    return {"final_result": response.content, "messages": [response]}

# --- 4. 그래프 구성 ---
workflow = StateGraph(ResearchState)

workflow.add_node("researcher", researcher)
workflow.add_node("reviewer", reviewer)
workflow.add_node("writer", writer)

# 순차적 흐름: Researcher -> Reviewer -> Writer
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "reviewer")
workflow.add_edge("reviewer", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

# --- 5. 실행 ---
if __name__ == "__main__":
    try:
        print("--- Multi-Agent Collaboration 실행 ---")
        task = "양자 컴퓨터(Quantum Computing)의 현재 기술 수준과 미래 전망"
        inputs = {"task": task, "messages": [HumanMessage(content=task)]}
        
        result = app.invoke(inputs)
        
        print("\n=== 최종 보고서 ===")
        print(result["final_result"])
        
    except Exception as e:
        print(f"Error: {e}")

