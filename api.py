from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Json
from typing import List
import uvicorn
import agent
import os
from agent import Agento
from dotenv import load_dotenv
import logging
from typing import Dict, Any
import yaml
import re
from replacement_chains import Agent


CANNED_RESPONSES=False

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv


def establish_connection():
    AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"

    agent = Agento(AGENT_NAME)
    agent.createIndex()
    return agent
load_dotenv()


app = FastAPI(debug=True)

class Payload(BaseModel):
    payload: Dict[str, Any]

class ImageResponse(BaseModel):
    success: bool
    message: str

import json

@app.post("/variate-assumption", response_model=dict)
async def variate_assumption(request_data: Payload) -> dict:

    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(json_payload['variate_assumption']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/variate-diet-assumption", response_model=dict)
async def variate_diet_assumption(request_data: Payload) -> dict:

    json_payload = request_data.payload
    agent_instance =Agent()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.update_agent_preferences(str(json_payload['variate_assumption']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/variate-goal", response_model=dict)
async def variate_goal(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(json_payload['variate_goal']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/data-request", response_model=dict)
async def data_request(request_data: Payload) -> dict:
    if CANNED_RESPONSES:
        with open('fixtures/recipe_response.json', 'r') as f:
            json_data = json.load(f)
            stripped_string_dict = {"response": json_data}

            # Return a JSON response with the new dictionary
            return JSONResponse(content=stripped_string_dict)

    with open('prompts.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    json_payload = request_data.payload
    default_query = data['default_query']

    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    i = 0;
    for key, val in factors_dict.items():
        i = i + 1
        default_query = default_query.replace('{factor%s}' % i, str("%s: %s" % (key, val)))
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
    stripped_string_dict = {"response": json.loads(stripped_string_output)}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)


@app.post("/recipe-request", response_model=dict)
async def recipe_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.solution_generation(factors_dict)
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
    if CANNED_RESPONSES:
        with open('fixtures/goal_response.json', 'r') as f:
            json_data = json.load(f)
            stripped_string_dict = {"response": json_data}

            # Return a JSON response with the new dictionary
            return JSONResponse(content=stripped_string_dict)

    with open('prompts.yaml', 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    default_query = data['optimize_goal']
    json_payload = request_data.payload
    agent_instance = establish_connection()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.action(str(default_query))
    print("---------------->>>>")
    print(output)
    print("---------------->>>>>")
    start = output.find('{')
    end = output.rfind('}')

    if start != -1 and end != -1:
        stripped_string_output = output[start:end + 1]
        print(stripped_string_output)
    else:
        print("No JSON data found in string.")
    stripped_string_dict = {"response": json.loads(stripped_string_output)}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)

@app.post("/optimize-diet-goal", response_model=dict)
async def optimize_diet_goal(request_data: Payload) -> dict:
    json_payload = request_data.payload
    # factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    # agent_instance = establish_connection()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.goal_optimization({})
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
