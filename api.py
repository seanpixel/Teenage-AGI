from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import os

from dotenv import load_dotenv
import logging
from typing import Dict, Any
from replacement_chains import Agent


CANNED_RESPONSES=False

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Set the log message format
)

logger = logging.getLogger(__name__)
from dotenv import load_dotenv


# def establish_connection():
#     AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"
#
#     agent = Agento(AGENT_NAME)
#     agent.createIndex()
#     return agent
load_dotenv()


app = FastAPI(debug=True)

class Payload(BaseModel):
    payload: Dict[str, Any]

class ImageResponse(BaseModel):
    success: bool
    message: str


@app.post("/variate-diet-assumption", response_model=dict)
async def variate_diet_assumption(request_data: Payload) -> dict:

    json_payload = request_data.payload
    agent_instance =Agent()
    agent_instance.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent_instance.update_agent_preferences(str(json_payload['variate_assumption']))
    stripped_string_dict = {"response": output}

    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)



@app.post("/variate-food-goal", response_model=dict)
async def variate_food_goal(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])

    output = agent.goal_optimization(str(json_payload['variate_goal']), model_speed="slow")
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

@app.post("/recipe-request", response_model=dict)
async def recipe_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.recipe_generation(factors_dict, model_speed="slow")
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

@app.post("/restaurant-request", response_model=dict)
async def restaurant_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.restaurant_generation(factors_dict, model_speed="slow")
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


@app.post("/delivery-request", response_model=dict)
async def delivery_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = await agent.delivery_generation(factors_dict, zipcode=json_payload["zipcode"], model_speed="slow")
    print("HERE IS THE OUTPUT", output)
    stripped_string_dict = {"response": output}
    # Return a JSON response with the new dictionary
    return JSONResponse(content=stripped_string_dict)
@app.post("/solution-request", response_model=dict)
async def solution_request(request_data: Payload) -> dict:
    json_payload = request_data.payload
    factors_dict = {factor['name']: factor['amount'] for factor in json_payload['factors']}
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.solution_generation(factors_dict, model_speed=json_payload["model_speed"])
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
@app.post("/generate-diet-goal", response_model=dict)
async def generate_diet_goal(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.goal_generation({}, model_speed= json_payload["model_speed"])
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

@app.post("/generate-diet-sub-goal", response_model=dict)
async def generate_diet_sub_goal(request_data: Payload) -> dict:
    json_payload = request_data.payload
    agent = Agent()
    agent.set_user_session(json_payload["user_id"], json_payload["session_id"])
    output = agent.sub_goal_generation(factors=json_payload["factors"], model_speed= json_payload["model_speed"])
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
