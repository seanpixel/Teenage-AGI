from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Json
from typing import List
import uvicorn
import agent
import os
from agent import Agent
from dotenv import load_dotenv
import logging
from typing import Dict, Any
import yaml
# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
# import agent
from dotenv import load_dotenv
# from main import agent

def establish_connection():
    AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"

    agent = Agent(AGENT_NAME)
    agent.createIndex()
    return agent
load_dotenv()

# Creates Pinecone Index
# agent.createIndex()

# Load default environment variables (.env)

app = FastAPI(debug=True)

class Payload(BaseModel):
    payload: Dict[str, Any]

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
import json
@app.post("/data-request", response_model=dict)
async def data_request(request_data: Payload) -> dict:
    with open('prompts.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    json_payload = request_data.payload
    default_query = data['default_query']
    default_query = default_query.replace("{factor_1}", json_payload["factor_1"]).replace("{factor_2}",
                                                                                    json_payload["factor_2"]).replace(
        "{factor_3}", json_payload["factor_3"]).replace("{factor_2_option}",
                                                                                    json_payload["factor_2_option"] ).replace(
        "{factor_3_option}", json_payload["factor_3_option"])
    print("here is the request data", json_payload["query"])
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    # # response_data = process_request_data(request_data.dict())
    # q = json.loads(request_data.query)
    # logging.info("HERE IS THE QUERY", q)
    # query = q.get("query")
    output = agent_instance.action(str(default_query))
    return {"response": output}
def start_api_server():
    # agent = establish_connection()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api_server()
