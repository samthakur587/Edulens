from fastapi import FastAPI
from app.tasks import process_video
from pydantic import BaseModel

app = FastAPI()

class VideoProcessRequest(BaseModel):
    video_url: str

@app.post("/process_video")
async def process_video_endpoint(request: VideoProcessRequest):
    task = process_video.delay(request.video_url)
    return {"task_id": task.id}
meta_data = None
@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    task = process_video.AsyncResult(task_id)
    global meta_data
    meta_data = task.result
    return {"status": task.status, "result": task.result}

print(f"bhai meta_data bhi aa rha h yha se embadding start kr de {meta_data}")