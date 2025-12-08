from fastapi import FastAPI
from dotenv import load_dotenv
from router import router

# 환경 변수 로드 (.env 파일)
load_dotenv()

app = FastAPI(title="RAG API 서비스", description="RAG 기반 PDF 채팅을 위한 API")

# 라우터 등록
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
