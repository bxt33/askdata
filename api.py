"""FastAPI backend for Shuxi analysis workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from storage import APP_DIR, AppStorage
from task_queue import TaskQueue
from utils import DataframeAgentFacade


app = FastAPI(title="Shuxi Analysis API", version="1.0.0")
storage = AppStorage()
queue = TaskQueue()


class AnalyzeRequest(BaseModel):
    dataset_id: str
    query: str
    human_confirmed: bool = False
    provider: str = "deepseek"
    api_key: str = ""
    model_name: str = "deepseek-v4-flash"


@app.on_event("startup")
def seed_samples() -> None:
    storage.seed_sample_dataset("house_price.csv", APP_DIR / "house_price.csv")
    storage.seed_sample_dataset("personal_data.csv", APP_DIR / "personal_data.csv")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "queue": queue.mode}


@app.get("/datasets")
def datasets() -> Dict[str, Any]:
    return {"datasets": storage.list_datasets()}


@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)) -> Dict[str, str]:
    if not file.filename or Path(file.filename).suffix.lower() not in {".csv", ".pdf", ".docx", ".doc"}:
        raise HTTPException(status_code=400, detail="Only CSV, PDF, DOCX, and DOC files are supported")
    dataset_id = storage.save_uploaded_dataset(file.filename, await file.read())
    return {"dataset_id": dataset_id}


def run_analysis_task(task_id: str, req: AnalyzeRequest) -> None:
    try:
        storage.update_task(task_id, "running", 20)
        df = storage.read_dataset(req.dataset_id)
        api_key = req.api_key or os.getenv("DEEPSEEK_API_KEY", "")
        facade = DataframeAgentFacade(api_key=api_key, provider=req.provider, model_name=req.model_name)
        storage.update_task(task_id, "running", 60)
        result = facade.analyze(df, req.query, human_confirmed=req.human_confirmed)
        storage.update_task(task_id, "succeeded", 100, result=result)
    except Exception as exc:
        storage.update_task(task_id, "failed", 100, error=str(exc))


@app.post("/analyze")
def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks) -> Dict[str, str]:
    if not storage.get_dataset(req.dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    task_id = storage.create_task(req.query, req.dataset_id)
    if queue.mode == "rq":
        job_id = queue.enqueue(run_analysis_task, task_id, req)
    else:
        background_tasks.add_task(run_analysis_task, task_id, req)
        job_id = "fastapi-background"
    return {"task_id": task_id, "queue": queue.mode, "job_id": job_id}


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    return {"tasks": storage.list_tasks()}


@app.get("/tasks/{task_id}")
def task(task_id: str) -> Dict[str, Any]:
    item = storage.get_task(task_id)
    if not item:
        raise HTTPException(status_code=404, detail="Task not found")
    return item
