from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# 1. 원하는 출력 구조 정의 (Pydantic)
class Joke(BaseModel):
    setup: str = Field(description="농담의 빌드업 부분")
    punchline: str = Field(description="농담의 핵심 웃음 포인트")

class JokeList(BaseModel):
    jokes: List[Joke] = Field(description="농담 리스트")

# 2. 파서 초기화
parser = PydanticOutputParser(pydantic_object=JokeList)

# 3. 프롬프트 정의 (포맷 지시사항 포함)
prompt = ChatPromptTemplate.from_messages([
    ("system", "당신은 유머 감각이 뛰어난 코미디언입니다.\n{format_instructions}"),
    ("user", "{topic}에 대한 농담을 3개 만들어주세요."),
])

# 4. 모델 설정
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# 5. 체인 연결
# partial_variables를 사용하여 format_instructions를 미리 주입
chain = prompt.partial(format_instructions=parser.get_format_instructions()) | model | parser

# 6. 실행
try:
    result = chain.invoke({"topic": "AI"})
    
    print(f"--- 생성된 농담 ({len(result.jokes)}개) ---")
    for idx, joke in enumerate(result.jokes, 1):
        print(f"{idx}. {joke.setup} -> {joke.punchline}")
        
except Exception as e:
    print(f"Error: {e}")

