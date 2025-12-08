import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 모델 설정
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 1. 첫 번째 체인: 주제 -> 아웃라인 생성
outline_prompt = ChatPromptTemplate.from_template("다음 주제에 대한 블로그 글의 목차(아웃라인)를 작성해줘: {topic}")
outline_chain = outline_prompt | model | StrOutputParser()

# 2. 두 번째 체인: 아웃라인 -> 본문 작성
article_prompt = ChatPromptTemplate.from_template(
    """
    다음 목차를 기반으로 블로그 글의 첫 번째 섹션만 작성해줘.
    
    [목차]
    {outline}
    """
)
article_chain = article_prompt | model | StrOutputParser()

# 3. 체인 연결 (Sequential Chain)
# 첫 번째 체인의 출력이 두 번째 체인의 입력으로 전달됨
full_chain = ({"outline": outline_chain} | article_chain)

try:
    print("--- Sequential Chain 실행 ---")
    result = full_chain.invoke({"topic": "생성형 AI의 미래"})
    print(result)
except Exception as e:
    print(f"Error: {e}")

