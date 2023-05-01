# PromethAI

PromethAI is a Python-based AGI (artificial general intelligence) project that recommends food choices based on a user's goals and preferences, and can modify its recommendations based on user feedback.
The project is built on top of an existing AGI project, and redone using Langchain library that uses OpenAI and Pinecone to give memory to the AI agent, and allows it to "think" before making an action (outputting text).

In the modified version, which we can call "PromethAI", the original project has been adapted to focus on food recommendations. The AI agent now has the ability to suggest meal options based on a user's specified goal, 
such as a fast meal, a tasty meal, or a healthy meal. It also has the ability to retrieve a list of restaurants from Google Maps and suggest matching food options based on the user's preferences.

Overall, PromethAI is a practical application of AGI technology that has the potential to help users make informed food choices based on their goals and preferences.

Credits: 
Teenage AGI -> https://github.com/seanpixel/Teenage-AGI
Baby AGI -> https://github.com/yoheinakajima/babyagi


## Objective
Inspired by the several Auto-GPT related Projects (predominently BabyAGI) and the Paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442), the original python project uses OpenAI and Pinecone to Give memory to an AI agent and also allows it to "think" before making an action (outputting text). 


## Quick start 
API_ENABLED switches between API and version that works using commandline 

```docker-compose build --build-arg API_ENABLED=True teenage-agi```



## How it Works
Here is what happens everytime the AI is queried by the user:
1. AI vectorizes the query and stores it in a Pinecone Vector Database
2. AI looks inside its memory and finds memories and past queries that are relevant to the current query
3. AI thinks about what action to take
4. AI stores the thought from Step 3
5. Based on the thought from Step 3 and relevant memories from Step 2, AI generates an output
6. AI stores the current query and its answer in its Pinecone vector database memory

## How to Use
1. Clone the repository via `git clone https://github.com/seanpixel/Teenage-AGI.git` and cd into the cloned repository.
2. Install required packages by doing: pip install -r requirements.txt
3. Create a .env file from the template `cp .env.template .env`
4. `open .env` and set your OpenAI and Pinecone API, Google Maps API key and Replicate API token
5. Run the script by launching a docker container with
```
docker-compose build --build-arg API_ENABLED=True teenage-agi
```
6. Access the API by doing CURL requests, example: 
```
curl -X POST "http://0.0.0.0:8000/data-request" -H "Content-Type: application/json" --data-raw 

```
## List of available endpoints

The available endpoints in the provided code are:
```
POST request to '/variate-diet-assumption' endpoint that takes a JSON payload containing 'user_id', 'session_id' and 'variate_assumption' keys, and returns a JSON response with a 'response' key.
POST request to '/variate-food-goal' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors', and 'variate_goal' keys, and returns a JSON response with a 'response' key.
POST request to '/recipe-request' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors' keys, and returns a JSON response with a 'response' key.
POST request to '/restaurant-request' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors' keys, and returns a JSON response with a 'response' key.
POST request to '/delivery-request' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors', and 'zipcode' keys, and returns a JSON response with a 'response' key.
POST request to '/solution-request' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors', and 'model_speed' keys, and returns a JSON response with a 'response' key.
POST request to '/generate-diet-goal' endpoint that takes a JSON payload containing 'user_id', 'session_id', and 'model_speed' keys, and returns a JSON response with a 'response' key.
POST request to '/generate-diet-sub-goal' endpoint that takes a JSON payload containing 'user_id', 'session_id', 'factors', and 'model_speed' keys, and returns a JSON response with a 'response' key.
```
All endpoints receive a payload in JSON format and return a response in JSON format.

Example of curl requests
```
curl --location --request POST 'http://0.0.0.0:8000/recipe-request' \
--header 'Content-Type: application/json' \
--data-raw '{
  "payload": {
    "user_id": "657",
    "session_id": "456",
    "model_speed":"slow",
    "factors": [
      {
        "name": "time",
        "amount": 90
      },
      {
        "name": "cost",
        "amount": 50
      },
      {
        "name": "health",
        "amount": 95
      }
    ]
  }
}'
```

```
curl --location --request POST 'http://0.0.0.0:8000/generate-diet-goal' \
--header 'Content-Type: application/json' \
--data-raw '{"payload": {"user_id": "658", "session_id": "457", "model_speed":"slow"}}'
```
```
curl --location --request POST 'http://0.0.0.0:8000/variate-assumption' \
--header 'Content-Type: application/json' \
--data-raw '{"payload": {"user_id": "657", "session_id": "456", "variate_assumption": "Remove {{Assumption}} from the list of assumptions"}}'
```

```
curl --location --request POST 'http://0.0.0.0:8000/generate-diet-sub-goal' \
--header 'Content-Type: application/json' \
--data-raw '{
  "payload": {
    "user_id": "657",
    "session_id": "456",
    "model_speed": "slow",
    "factors": [
      {
        "name": "Portion Control"
      },
      {
        "name": "Cuisine"
      },
      {
        "name": "Macronutrients"
      }
    ]
  }
}'
```
