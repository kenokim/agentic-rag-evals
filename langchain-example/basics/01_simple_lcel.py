import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 모델 설정 (Google Gemini)
# os.environ["GOOGLE_API_KEY"] = "AIza..."
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. 프롬프트 템플릿 생성
prompt = ChatPromptTemplate.from_template("다음 주제에 대해 짧은 농담을 해줘: {topic}")

# 3. 출력 파서 설정
output_parser = StrOutputParser()

# 4. LCEL(LangChain Expression Language) 체인 구성
# prompt | model | output_parser 순서로 데이터가 흐릅니다.
chain = prompt | model | output_parser

# 5. 체인 실행
try:
    result = chain.invoke({"topic": "개발자"})
    print("--- 결과 ---")
    print(result)
except Exception as e:
    print(f"Error: {e}")

