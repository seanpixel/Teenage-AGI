from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
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
import re
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
@app.post("/variate-assumption", response_model=dict)
async def data_request(request_data: Payload) -> dict:

    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(json_payload['variate_assumption']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)
@app.post("/variate-goal", response_model=dict)
async def data_request(request_data: Payload) -> dict:

    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(json_payload['variate_goal']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/data-request", response_model=dict)
async def data_request(request_data: Payload) -> dict:
    with open('prompts.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    json_payload = request_data.payload
    default_query = data['default_query']

    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    template_vals = list(set(re.findall(r'(\{\w+\})', default_query)))
    filtered_template_vals = [x for x in template_vals if "value" not in x]
    filtered_template_vals_def = [x for x in template_vals if "value"  in x]
    logging.info("HERE ARE THE TEMPLATE VALS", str(filtered_template_vals))
    for key, val in factors_dict.items():
        for value in filtered_template_vals:
            if key in value:
                default_query = default_query.replace(value, key)
                for amount_value in filtered_template_vals_def:
                    if key in amount_value:
                        default_query = default_query.replace(amount_value, str(val))
    logging.info("HERE STARTS THE DEFAULT QUERY TEMPLATED FOR DEBUGGING PURPOSES", str(default_query))

    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(default_query))
    start = output.find('{')
    end = output.rfind('}')

    if start != -1 and end != -1:
        stripped_string_output = output[start:end + 1]
        print(stripped_string_output)
    else:
        print("No JSON data found in string.")
    stripped_string_dict = {"response": stripped_string_output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/optimize-goal", response_model=dict)
async def data_request(request_data: Payload) -> dict:
    with open('prompts.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    default_query = data['optimize_goal']
    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(default_query))
    start = output.find('{')
    end = output.rfind('}')

    if start != -1 and end != -1:
        stripped_string_output = output[start:end + 1]
        print(stripped_string_output)
    else:
        print("No JSON data found in string.")
    stripped_string_dict = {"response": stripped_string_output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)
def start_api_server():
    # agent = establish_connection()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start_api_server()
