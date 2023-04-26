import agent
import os
from agent import Agento
from dotenv import load_dotenv
from api import start_api_server
import importlib
import uvicorn
API_ENABLED = os.environ.get("API_ENABLED", "False").lower() == "true"

# Load default environment variables (.env)
load_dotenv()

def can_import(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

AGENT_NAME = os.getenv("AGENT_NAME") or "my-agent"

agent = Agento(AGENT_NAME)

# Creates Pinecone Index
agent.createIndex()
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

# Model configuration
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))


DOTENV_EXTENSIONS = os.getenv("DOTENV_EXTENSIONS", "").split(" ")

# Command line arguments extension
# Can override any of the above environment variables
ENABLE_COMMAND_LINE_ARGS = (
    os.getenv("ENABLE_COMMAND_LINE_ARGS", "false").lower() == "true"
)
if ENABLE_COMMAND_LINE_ARGS:
    if can_import("extensions.argparseext"):
        from extensions.argparseext import parse_arguments

        OBJECTIVE, INITIAL_TASK, OPENAI_API_MODEL, DOTENV_EXTENSIONS = parse_arguments()

# Load additional environment variables for enabled extensions
if DOTENV_EXTENSIONS:
    if can_import("extensions.dotenvext"):
        from extensions.dotenvext import load_dotenv_extensions

        load_dotenv_extensions(DOTENV_EXTENSIONS)
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
