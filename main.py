import agent
from agent import Agent

agent = Agent("agent2")

agent.createIndex()

print("Talk to the AI!")

while True:
    userInput = input()
    print("\n", agent.action(userInput))
