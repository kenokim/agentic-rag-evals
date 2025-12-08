from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# 1. 도구 정의
@tool
def get_weather(city: str) -> str:
    """Get the weather for a specific city."""
    # 실제 API 호출 대신 모의 응답
    if "Seoul" in city:
        return "Seoul is currently 10 degrees Celsius and sunny."
    elif "London" in city:
        return "London is currently 5 degrees Celsius and rainy."
    return "Unknown city."

tools = [get_weather]

# 2. LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 3. 프롬프트 로드 (LangChain Hub에서 표준 프롬프트 가져오기)
# hwchase17/openai-tools-agent는 Tool Calling Agent를 위한 기본 프롬프트입니다.
prompt = hub.pull("hwchase17/openai-tools-agent")

# 4. Agent 생성 (Legacy 방식 - LangGraph 사용 권장하지만, AgentExecutor도 여전히 사용됨)
agent = create_tool_calling_agent(llm, tools, prompt)

# 5. AgentExecutor 생성
# verbose=True로 설정하면 사고 과정(Thought Process)을 볼 수 있습니다.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 실행
try:
    print("--- Tool Calling Agent 실행 ---")
    result = agent_executor.invoke({"input": "What is the weather in Seoul?"})
    print(f"답변: {result['output']}")
except Exception as e:
    print(f"Error: {e}")

