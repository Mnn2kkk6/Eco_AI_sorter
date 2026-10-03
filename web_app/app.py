from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
load_dotenv()
from src.inference import predict
from web_app import agent

BASE = Path(__file__).resolve().parent
app = FastAPI(title="EcoAI Sorter")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE / "templates" / "ecoai_sorter.html")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Chỉ nhận file ảnh")
    try:
        return predict(await file.read())
    except FileNotFoundError:
        raise HTTPException(503, "Chưa có trọng số mô hình trong weights/ — hãy train trước")


class Msg(BaseModel):
    role: str
    content: str = Field(max_length=4000)


class AgentReq(BaseModel):
    messages: List[Msg] = Field(max_length=20)
    context: Optional[dict] = None


@app.get("/api/agent/status")
def agent_status():
    return agent.status()


@app.post("/api/agent/chat")
def agent_chat(req: AgentReq):
    try:
        return agent.run_agent([m.dict() for m in req.messages], req.context)
    except agent.AgentError as e:
        raise HTTPException(503, str(e))