from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
import uvicorn
import agent
import os
from agent import Agent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME") or "api-agent"

agent = Agent(AGENT_NAME)
app = FastAPI()

class Payload(BaseModel):
    user_id: str
    session_id: str

class ImageResponse(BaseModel):
    success: bool
    message: str

# @app.post("/process-payload", response_model=Payload)
# async def process_payload(payload: Payload) -> Payload:
#     value = calculate_value(payload.user_id, payload.session_id)
#     payload.dict().update({"value": value})
#     return payload
#
# @app.post("/upload-images", response_model=ImageResponse)
# async def upload_images(files: List[UploadFile] = File(...)) -> ImageResponse:
#     success = process_images(files)
#     message = "Images uploaded and processed successfully." if success else "Failed to process images."
#     return ImageResponse(success=success, message=message)

@app.post("/data-request", response_model=dict)
async def data_request(request_data: Payload) -> dict:
    response_data = process_request_data(request_data.dict())
    response_data.update({
        "user_id": request_data.user_id,
        "session_id": request_data.session_id
    })

    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)