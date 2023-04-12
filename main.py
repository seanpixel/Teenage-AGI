import agent
from agent import Agent

agent = Agent("myAgent")

# Creates Pinecone Index
agent.createIndex()

print("Talk to the AI!")

while True:
    userInput = input()
    print("\n", agent.action(userInput))
