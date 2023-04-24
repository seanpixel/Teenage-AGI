from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.chains import LLMSummarizationCheckerChain
import pinecone
import re
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import openai
import datetime
from jinja2 import Template
import time
import os
import yaml

from langchain.agents import (
    create_json_agent,
    AgentExecutor
)
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.requests import TextRequestsWrapper
from langchain.tools.json.tool import JsonSpec
from pydantic import BaseModel, Field

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
from heuristic_experience_orchestrator.prompt_template_modification import PromptTemplate
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document
from langchain.vectorstores import FAISS, Pinecone
from heuristic_experience_orchestrator.task_identification import TaskIdentificationChain

import os
# nltk.download('punkt')
import subprocess

# database_url = os.environ.get('DATABASE_URL')
# import nltk


OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"

OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")

assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"


class Agent():
    def __init__(self, table_name=None, user_id: Optional[str] = None, session_id: Optional[str] = None) -> None:
        self.table_name = table_name
        self.user_id = user_id
        self.session_id = session_id
        self.memory = None
        self.thought_id_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # Timestamp with millisecond precision
        self.last_message = ""
        self.llm =  OpenAI(temperature=0.0,openai_api_key = OPENAI_API_KEY)
        self.verbose: bool = True
        self.openai_model = "gpt-3.5-turbo"
        self.openai_temperature = 0.0

    def openai_call(self,
            prompt: str,
            model: str = OPENAI_MODEL,
            temperature: float = OPENAI_TEMPERATURE,
            max_tokens: int = 2000,
    ):
        while True:
            try:
                if model.startswith("llama"):
                    # Spawn a subprocess to run llama.cpp
                    cmd = ["llama/main", "-p", prompt]
                    result = subprocess.run(cmd, shell=True, stderr=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            text=True)
                    return result.stdout.strip()
                else:
                    # Use chat completion API
                    messages = [
                        {"role": "system",
                         "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
                        {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
                        {"role": "user", "content": prompt},
                    ]
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        stop=None,
                    )
                    return response.choices[0].message.content
            except openai.error.RateLimitError:
                print(
                    "The OpenAI API rate limit has been exceeded. Waiting 10 seconds and trying again."
                )
                time.sleep(10)  # Wait 10 seconds and try again
            else:
                break
    def set_user_session(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]

    def init_pinecone(self):
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
            index_name = "my-agent"
            self.memory = pinecone.Index(index_name)

    # this is code the get data from pinecone with similarity search, should be redone

    # def create_new_memory_retriever():
    #     """Create a new vector store retriever unique to the agent."""
    #     # Define your embedding model
    #     embeddings_model = OpenAIEmbeddings(openai_api_key = "")
    #     # Initialize the vectorstore as empty

    #     pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    #     index_name = "my-agent"
    #     embeddings = OpenAIEmbeddings(openai_api_key = ")
    #     embedding_size = 1536
    #     # index = faiss.IndexFlatL2(embedding_size)
    #     vectorstore = Pinecone.from_existing_index( index_name, embeddings)
    #     return TimeWeightedVectorStoreRetriever(vectorstore=vectorstore, decay_rate=.0000000000000000000000001, k=15)

    # aa = create_new_memory_retriever()
    #
    # query = "What did the president say about Ketanji Brown Jackson"
    # docs = aa.get_relevant_documents(query)
    # memory_retriever= TimeWeightedVectorStoreRetriever

    # def fetch_memories(self, observation: str) -> List[Document]:
    #     """Fetch related memories."""
    #     return self.memory_retriever.get_relevant_documents(observation)
    #

    #
    # prompt = PromptTemplate(
    #     input_variables=["product"],
    #     template="What is a good name for a company that makes {product}?",
    # )
    #
    def _update_memories(self, observation: str, namespace: str):
        # Fetch related characteristics
        vector = self.get_ada_embedding(observation)
        upsert_response = self.memory.upsert(
            vectors=[
                {
                    'id': f"thought-{self.thought_id_timestamp}",
                    'values': vector,
                    'metadata':
                        {"thought_string": observation, "user_id": self.user_id
                         }
                }],
            namespace=namespace,
        )

    def _fetch_memories(self, observation: str, namespace:str) -> List[Document]:
          #"""Fetch related characteristics, preferences or dislikes for a user."""
        query_embedding = self.get_ada_embedding(observation)
        self.memory.query(query_embedding, top_k=1, include_metadata=True, namespace=namespace,
                          filter={'user_id': {'$eq': self.user_id}})
    #     return self.memory_retriever.get_relevant_documents(observation)
    def _compute_agent_summary(self):
        """Computes summary for a person"""
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_characteristics}"
            + "{relevant_preferences}"
            + "{relevant_dislikes}"
            + "Do not embellish."
            + "\n\nSummary: "
        )
        self.init_pinecone()
        # The agent seeks to think about their core characteristics.
        relevant_characteristics = self._fetch_memories(f"Users core characteristics", namespace="TRAITS")
        relevant_preferences = self._fetch_memories(f"Users core preferences", namespace="PREFERENCES")
        relevant_dislikes = self._fetch_memories(f"Users core dislikes", namespace="DISLIKES")
        chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)

        return chain.run(name= self.user_id, relevant_characteristics=relevant_characteristics, relevant_preferences=relevant_preferences, relevant_dislikes=relevant_dislikes).strip()



    def update_agent_preferences(self, preferences:str):
        """Serves to update agents preferences so that they can be used in summary"""

        prompt = """ The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """
        self.init_pinecone()
        past_preference = self._fetch_memories(f"Users core preferences", namespace="PREFERENCES")
        prompt = PromptTemplate(input_variables=["name", "past_preference", "preferences"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_preference= past_preference, preferences=preferences)

        return self._update_memories(prompt, namespace="PREFERENCES")

    def update_agent_taboos(self, dislikes:str):
        """Serves to update agents taboos so that they can be used in summary"""
        prompt =""" The {name} has following {past_dislikes} and the new {dislikes}
                Update user taboos and return a list of taboos
            Do not embellish.
            Summary: """
        self.init_pinecone()
        past_dislikes = self._fetch_memories(f"Users core dislikes", namespace="DISLIKES")
        prompt = PromptTemplate(input_variables=["name", "past_preference", "preferences"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)

        return self._update_memories(prompt, namespace="DISLIKES")


    def update_agent_traits(self, traits:str):
        """Serves to update agent traits so that they can be used in summary"""
        prompt =""" The {name} has following {past_traits} and the new {traits}
                Update user taboos and return a list of taboos
            Do not embellish.
            Summary: """
        self.init_pinecone()
        past_traits = self._fetch_memories(f"Users core dislikes", namespace="DISLIKES")
        prompt = PromptTemplate(input_variables=["name", "past_traits", "traits"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_traits= past_traits, dislikes=traits)

        return self._update_memories(prompt, namespace="DISLIKES")


    def update_agent_summary(self):
        """Serves to update agent traits so that they can be used in summary"""
        summary = self._compute_agent_summary()
        return self._update_memories(summary, namespace="SUMMARY")

    def task_identification(self, goals:str):
        """Serves to update agent traits so that they can be used in summary"""
        self.init_pinecone()
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        complete_query = str(agent_summary) + goals
        complete_query = PromptTemplate.from_template(complete_query)
        print("HERE IS THE COMPLETE QUERY", complete_query)
        from heuristic_experience_orchestrator.task_identification import TaskIdentificationChain
        chain = TaskIdentificationChain.from_llm(llm=self.llm, task_description="none",  value="Decomposition", verbose=self.verbose)

        chain_output = chain.run(name= self.user_id).strip()
        return chain_output


    def solution_generation(self, factors:dict):
        """Serves to update agent traits so that they can be used in summary"""


        prompt = """
        Hey ChatGPT, I need your help choosing what to eat for my next meal.
        There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
        {% for factor, value in factors.items() %}
        For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
        {% endfor %}
        
        However, I want you to make six assumptions that would allow you to come up with one suggestion for me. 
      They should be very short, be very specific. Instructions and ingredients should not be shortened. I have cooking skills and access to basic cooking equipment, it should not be an assumption. 
      If flavors and cuisines are mentioned, one cuisine should be chosen based on ChatGPT preferences without stating it's ChatGPT favourite. 
      Answer with a JSON array that follows the following structure
      {
            "Result type": "Recipe",
            "Assumptions": ["Vegetarian", "Gluten-free", "Low-carb"],
            "Body": {
              "title": "Name of the result",
              "rating": 4.8,
              "image_link": "image link of the product",
              "prep_time": "prep time in minutes" ,
              "cook_time": "cook time in minutes" ,
              "description": "Product description",
              "ingredients": [
                "1 large ingredient",
                "2 other ingredieents"
              ],
              "instructions": [
                "Do something",
                "Do something else",
                "Do even more"
              ]
            }
          }
      """
        import json
        json_str = json.dumps( {
            "Result type": "Recipe",
            "Assumptions": ["Vegetarian", "Gluten-free", "Low-carb"],
            "Body": {
              "title": "Name of the result",
              "rating": 4.8,
              "image_link": "image link of the product",
              "prep_time": "prep time in minutes" ,
              "cook_time": "cook time in minutes" ,
              "description": "Product description",
              "ingredients": [
                "1 large ingredient",
                "2 other ingredieents"
              ],
              "instructions": [
                "Do something",
                "Do something else",
                "Do even more"
              ]
            }
          })

        input_data = [
            {"Recipe": json_str}
        ]

        # json_spec = JsonSpec(dict_=data, max_value_length=4000)
        # json_toolkit = JsonToolkit(spec=json_spec)
        #
        # json_agent_executor = create_json_agent(
        #     llm=self.llm,
        #     toolkit=json_toolkit,
        #     verbose=True
        # )
        self.init_pinecone()
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        from langchain.prompts.base import StringPromptTemplate

        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output

        chain_output = self.openai_call(complete_query)
        return chain_output
        # # complete_query = PromptTemplate.from_template(complete_query)
        # complete_query = PromptTemplate.from_template(complete_query)
        #
        # # json_agent_executor.run(complete_query)
        # chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        # aa = chain.run(input_data=input_data, name=self.user_id).strip()
        # return print(aa)

    def goal_optimization(self, factors: dict):
        """Serves to optimize agent goals"""

        prompt = """
              Based on all the history and information of this user, suggest three goals that are personal to him that he should apply to optimize his time.
              Only JSON values should be the output, don't write anything extra, no warnings no explanations. 
              Make sure to provide data in the following format
              {
                "Goals": [
                  {
                    "name": "Vegetarian",
                    "min": 0,
                    "max": 1,
                    "unit_name": "",
                    "option_array": [
                      "Yes",
                      "No"
                    ]
                  },
                  {
                    "name": "Gluten-free",
                    "min": 0,
                    "max": 1,
                    "unit_name": "",
                    "option_array": [
                      "Yes",
                      "No"
                    ]
                  },
                  {
                    "name": "Low-carb",
                    "min": 0,
                    "max": 100,
                    "unit_name": "g",
                    "option_array": []
                  }
                ]
              }
            """
        # json_spec = JsonSpec(dict_=data, max_value_length=4000)
        # json_toolkit = JsonToolkit(spec=json_spec)
        #
        # json_agent_executor = create_json_agent(
        #     llm=self.llm,
        #     toolkit=json_toolkit,
        #     verbose=True
        # )
        self.init_pinecone()
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        from langchain.prompts.base import StringPromptTemplate

        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output

        call_output = self.openai_call(complete_query)
        return call_output
        # # complete_query = PromptTemplate.from_template(complete_query)
        # complete_query = PromptTemplate.from_template(complete_query)
        #
        # # json_agent_executor.run(complete_query)
        # chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
        # aa = chain.run(input_data=input_data, name=self.user_id).strip()
        # return print(aa)

    def solution_evaluation_test(self):
        """Serves to update agent traits so that they can be used in summary"""
        return


    def solution_implementation(self):
        """Serves to update agent traits so that they can be used in summary"""
        return

    # checker_chain = LLMSummarizationCheckerChain(llm=llm, verbose=True, max_checks=2)
    # text = """
    # Your 9-year old might like these recent discoveries made by The James Webb Space Telescope (JWST):
    # • In 2023, The JWST spotted a number of galaxies nicknamed "green peas." They were given this name because they are small, round, and green, like peas.
    # • The telescope captured images of galaxies that are over 13 billion years old. This means that the light from these galaxies has been traveling for over 13 billion years to reach us.
    # • JWST took the very first pictures of a planet outside of our own solar system. These distant worlds are called "exoplanets." Exo means "from outside."
    # These discoveries can spark a child's imagination about the infinite wonders of the universe."""
    # checker_chain.run(text)


if __name__ == "__main__":
    agent = Agent()
    # agent.task_identification("I need your help choosing what to eat for my next meal. ")
    # agent.goal_optimization( {    'health': 85,
    # 'time': 75,
    # 'cost': 50,})