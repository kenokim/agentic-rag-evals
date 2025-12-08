# Agentic RAG Architecture

단순 RAG 파이프라인을 넘어, 스스로 판단하고 도구를 사용하는 **Agent** 구조로 시스템을 확장하는 방안을 정리합니다.

## 1. RAG vs Agentic RAG

*   **Standard RAG**: 질문 -> 검색(Retriever) -> 답변 생성. (선형적 구조, 무조건 검색 수행)
*   **Agentic RAG**: 질문 -> **계획/판단(Agent)** -> 도구 사용(검색, 계산, 웹 등) -> 답변 생성. (동적 구조, 필요할 때만 검색)

### 왜 Agent인가?
1.  **Routing**: 질문에 따라 "문서 검색"이 필요한지, "일반 대화"인지, "웹 검색"이 필요한지 스스로 판단합니다.
2.  **Self-Correction**: 검색 결과가 불충분하면 검색어를 수정해서 다시 검색(Re-writing)할 수 있습니다.
3.  **Multi-step Reasoning**: 복잡한 질문을 여러 단계로 쪼개서 정보를 수집하고 종합할 수 있습니다.

---

## 2. Tools Design (도구 설계)

Agent가 사용할 수 있는 도구들을 정의합니다. 가장 핵심적인 도구는 우리가 만든 "Retriever"입니다.

### Core Tools
1.  **Paper Retriever (필수)**
    *   **설명**: "업로드된 논문/PDF 문서에서 정보를 검색합니다."
    *   **기능**: 사용자의 질문과 관련된 문서 조각을 찾아옵니다.
    *   **구현**: 기존 RAG의 `vectorstore.as_retriever()`를 Tool 형태로 래핑.

2.  **Web Search (선택)**
    *   **설명**: "최신 정보나 논문에 없는 일반 상식을 검색합니다."
    *   **도구**: Tavily Search, Google Search 등.

3.  **Calculator (선택)**
    *   **설명**: "복잡한 수치 계산이 필요할 때 사용합니다."

---

## 3. Agent API Design

기존 Chat API를 확장하거나 새로운 엔드포인트를 추가하여 Agent 기능을 제공합니다.

### Endpoint Info
- **URL**: `/agent/chat`
- **Method**: `POST`

### Request Body
```json
{
  "query": "이 논문의 저자가 쓴 다른 논문의 내용을 웹에서 찾아보고 비교해줘.",
  "chat_history": [] 
}
```

### Response Body
Agent는 결론뿐만 아니라 **생각의 과정(Thought Process)**을 함께 반환하는 것이 디버깅과 사용자 경험에 좋습니다.

```json
{
  "answer": "이 논문의 저자는 ... (비교 내용) ... 입니다.",
  "steps": [
    {
      "tool": "Paper_Retriever",
      "tool_input": "논문 저자 이름",
      "tool_output": "Hong Gil Dong"
    },
    {
      "tool": "Web_Search",
      "tool_input": "Hong Gil Dong papers",
      "tool_output": "..."
    }
  ]
}
```

---

## 4. Implementation Strategy (LangGraph)

최신 LangChain 생태계에서는 순환형(Cyclic) 프로세스를 제어하기 위해 **LangGraph** 사용을 권장합니다.

### 상태(State) 정의
```python
from typing import TypedDict, Annotated, List, Union
from langchain_core.messages import BaseMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # 필요 시 추가 필드 (예: documents, query 등)
```

### 코드 예시 (Concept)

```python
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. 도구 생성
retriever_tool = create_retriever_tool(
    retriever,
    "paper_search",
    "Search for information about the uploaded paper."
)
tools = [retriever_tool]

# 2. LLM 설정 (Function Calling 지원 모델 권장)
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# 3. Agent 생성 (Prebuilt ReAct Agent 사용)
agent_executor = create_react_agent(llm, tools)

# 4. 실행
response = agent_executor.invoke({"messages": [("user", "이 논문의 핵심 아이디어는?")]})
print(response["messages"][-1].content)
```

### 고려사항
*   **Gemini Function Calling**: Gemini 모델은 Function Calling을 잘 지원하므로, Tool 사용에 적합합니다.
*   **Prompt Engineering**: Agent가 도구를 언제 써야 하는지 시스템 프롬프트에 명확히 지시해야 합니다. (예: "문서와 관련 없는 질문에는 도구를 쓰지 마시오.")

