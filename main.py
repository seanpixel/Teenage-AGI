import agent
import os
from agent import Agent
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"

agent = Agent(AGENT_NAME)

# Creates Pinecone Index
agent.createIndex()

print("Talk to the AI!")

while True:
    userInput = input()
    print("\n", agent.action(userInput))
