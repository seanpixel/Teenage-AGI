import agent
from agent import Agent

agent = Agent("agent2")

agent.createIndex()

#print("\n", agent.action("Hey, How are you doing?"))
print("Talk to the AI!")

while True:
    userInput = input()
    print("\n", agent.action(userInput))