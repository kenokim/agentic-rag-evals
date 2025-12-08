import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 간단한 챗 모델 사용 예제
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

messages = [
    SystemMessage(content="You are a helpful assistant who speaks like a pirate."),
    HumanMessage(content="Hello! How are you?"),
]

try:
    response = chat.invoke(messages)
    print("--- Pirate Chat ---")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")

