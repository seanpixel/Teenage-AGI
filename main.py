import agent
import os
from agent import Agent
from dotenv import load_dotenv
from api import start_api_server
import uvicorn
API_ENABLED = os.environ.get("API_ENABLED", "False").lower() == "true"

# Load default environment variables (.env)
load_dotenv()

# AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"
#
# agent = Agent(AGENT_NAME)
#
# # Creates Pinecone Index
# agent.createIndex()

print("Talk to the AI!")
if API_ENABLED:
    # Run FastAPI application
    start_api_server()

else:
    while True:
        userInput = input()
        if userInput:
            if (userInput.startswith("read:")):
                agent.read(" ".join(userInput.split(" ")[1:]))
                print("Understood! The information is stored in my memory.")
            elif (userInput.startswith("think:")):
                agent.think(" ".join(userInput.split(" ")[1:]))
                print("Understood! I stored that thought into my memory.")
            else:
                print(agent.action(userInput), "\n")
        else:
            print("SYSTEM - Give a valid input")
