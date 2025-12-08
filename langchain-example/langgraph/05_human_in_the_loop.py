import operator
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END

# --- 1. 모델 설정 ---
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# --- 2. 상태 정의 ---
class EditorState(TypedDict):
    """
    State for Human-in-the-loop workflow.
    - topic: 글 주제
    - draft: AI가 작성한 초안
    - feedback: 사람의 피드백 (없으면 승인으로 간주)
    - messages: 대화 기록 (옵션)
    """
    topic: str
    draft: str
    feedback: str

# --- 3. 노드(에이전트) 정의 ---

def writer(state: EditorState):
    """주제에 대해 초안을 작성하거나, 피드백을 반영하여 수정합니다."""
    print("--- Writer: 작성 중 ---")
    topic = state.get("topic")
    draft = state.get("draft")
    feedback = state.get("feedback")

    if draft and feedback:
        # 수정 모드
        prompt = ChatPromptTemplate.from_template(
            """
            다음 초안을 피드백을 반영하여 수정해주세요.
            
            [주제]: {topic}
            [초안]: {draft}
            [피드백]: {feedback}
            
            수정된 초안:
            """
        )
        response = (prompt | model).invoke({"topic": topic, "draft": draft, "feedback": feedback})
    else:
        # 신규 작성 모드
        prompt = ChatPromptTemplate.from_template("다음 주제에 대해 짧은 에세이 초안을 작성해줘: {topic}")
        response = (prompt | model).invoke({"topic": topic})

    return {"draft": response.content, "feedback": None} # 피드백 초기화

def human_review_node(state: EditorState):
    """
    (가상의) 사람 검토 단계
    실제 애플리케이션에서는 여기서 실행을 멈추고(interrupt),
    사용자 입력을 받은 후 resume 합니다.
    여기서는 코드로 시뮬레이션합니다.
    """
    pass # 실제 로직은 엣지 조건이나 외부 입력에서 처리됨

# --- 4. 조건부 엣지 ---
def review_router(state: EditorState) -> Literal["writer", "publisher"]:
    """
    사람의 피드백이 있으면 다시 writer로, 없으면(승인) publisher로 이동
    """
    feedback = state.get("feedback")
    if feedback:
        print(f"--- Reviewer: 피드백 발견 -> '{feedback}' ---")
        return "writer"
    
    print("--- Reviewer: 승인 완료 ---")
    return "publisher"

def publisher(state: EditorState):
    """최종 결과물 발행"""
    print("--- Publisher: 발행 완료! ---")
    print(f"최종 원고:\n{state['draft']}")
    return

# --- 5. 그래프 구성 ---
workflow = StateGraph(EditorState)

workflow.add_node("writer", writer)
workflow.add_node("human_review", human_review_node)
workflow.add_node("publisher", publisher)

workflow.add_edge(START, "writer")
workflow.add_edge("writer", "human_review")
workflow.add_conditional_edges(
    "human_review",
    review_router,
    {"writer": "writer", "publisher": "publisher"}
)
workflow.add_edge("publisher", END)

# 메모리(Checkpoint) 설정 - 실제 Human-in-the-loop에는 필수적이지만 여기서는 생략하고 직접 실행 흐름 제어
app = workflow.compile()

# --- 6. 실행 시뮬레이션 ---
if __name__ == "__main__":
    try:
        print("=== Human-in-the-loop 시뮬레이션 ===")
        
        # 1. 초기 실행 (초안 작성)
        initial_state = {"topic": "인공지능의 윤리", "draft": None, "feedback": None}
        print("\n[Step 1] 초안 작성 요청")
        # 실제로는 interrupt_before=["human_review"] 등을 사용하여 멈춰야 함
        # 여기서는 수동으로 상태를 조작하며 흐름을 보여줍니다.
        
        # Writer 실행
        state_after_write = writer(initial_state)
        print(f"\n[AI 초안]:\n{state_after_write['draft'][:100]}...") # 일부만 출력
        
        # 2. 사람이 검토 후 피드백 입력 (시나리오: 수정 요청)
        print("\n[Step 2] 사람 피드백 입력: '너무 길어. 좀 더 요약해줘.'")
        state_with_feedback = {**initial_state, **state_after_write, "feedback": "너무 길어. 좀 더 요약해줘."}
        
        # 피드백 반영하여 Writer 다시 실행 (Router가 writer로 보냄)
        # review_router 로직 테스트
        next_step = review_router(state_with_feedback) # -> writer
        if next_step == "writer":
            state_after_revise = writer(state_with_feedback)
            print(f"\n[AI 수정안]:\n{state_after_revise['draft'][:100]}...")
            
            # 3. 사람이 검토 후 승인 (시나리오: 피드백 없음)
            print("\n[Step 3] 사람 승인 (피드백 없음)")
            state_approved = {**state_with_feedback, **state_after_revise, "feedback": None}
            
            next_step_2 = review_router(state_approved) # -> publisher
            if next_step_2 == "publisher":
                publisher(state_approved)

    except Exception as e:
        print(f"Error: {e}")

