from fastapi import FastAPI
from api.routers import api
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 origin을 지정해야 합니다
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"message": "🚀🚀 열정 4팀 화이팅!!!!!"}

app.include_router(api.router, prefix="/api")

# "/static" URL 경로를 실제 디렉터리 "/home/ubuntu/fastapi/static"에 연결
app.mount("/static", StaticFiles(directory="/home/ubuntu/fastapi/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)