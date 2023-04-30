from langchain.prompts import PromptTemplate

import pinecone
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
import openai
import datetime
from jinja2 import Template
from dotenv import load_dotenv
import time
from langchain.llms.openai import OpenAI
from langchain import LLMChain
#from heuristic_experience_orchestrator.prompt_template_modification import PromptTemplate
# from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.schema import BaseLanguageModel, Document
import os
from food_scrapers import wolt_tool

from langchain.tools import GooglePlacesTool

# nltk.download('punkt')
import subprocess

# database_url = os.environ.get('DATABASE_URL')
# import nltk
load_dotenv()
from langchain.llms import Replicate
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-3.5-turbo"
GPLACES_API_KEY = "AIzaSyCAvRAf1eCJ27fTfjJauVgdhI5fodAFA_k"
OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", 0.0))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"
import os
os.environ["GPLACES_API_KEY"] = os.getenv("GPLACES_API_KEY", "")

class Agent():
    def __init__(self, table_name=None, user_id: Optional[str] = "user123", session_id: Optional[str] = None) -> None:
        self.table_name = table_name
        self.user_id = user_id
        self.session_id = session_id
        self.memory = None
        self.thought_id_timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:-3]  # Timestamp with millisecond precision
        self.last_message = ""
        self.llm = OpenAI(temperature=0.0,max_tokens = 1000, openai_api_key = OPENAI_API_KEY)
        self.replicate_llm = Replicate(model="replicate/vicuna-13b:a68b84083b703ab3d5fbf31b6e25f16be2988e4c3e21fe79c2ff1c18b99e61c1")
        self.verbose: bool = True
        self.openai_model = "gpt-3.5-turbo"
        self.openai_temperature = 0.0
        self.index = "my-agent"

    def test_replicate(self):
        start_time = time.time()
        bb = self.replicate_llm("""             Help me choose what food choice, order, restaurant or a recipe to eat or make for my next meal.     
                There are 'health', 'time', 'cost' factors I want to consider.
                
                For 'health', I want the meal to be '85' points on a scale of 1 to 100 points.
                
                For 'time', I want the meal to be '75' points on a scale of 1 to 100 points.
                
                For 'cost', I want the meal to be '50' points on a scale of 1 to 100 points.
                
                Instructions and ingredients should be detailed.  Result type can be Recipe, but not Meal
                Answer with a result in a correct  python dictionary that is properly formatted that contains the following keys and must have  values
                "Result type",  "body" which should contain "title", "rating", "prep_time", "cook_time", "description", "ingredients", "instructions" 
                The values in JSON should not repeat
                """)
        time.sleep(15)
        end_time = time.time()

        execution_time = end_time - start_time
        print("Execution time: ", execution_time, " seconds")
        return print(bb)
    def set_user_session(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id

    def get_ada_embedding(self, text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002",api_key =OPENAI_API_KEY)[
            "data"
        ][0]["embedding"]

    def init_pinecone(self, index_name):
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
            return pinecone.Index(index_name)
    def _update_memories(self, observation: str, namespace: str):
        # Fetch related characteristics
        memory = self.init_pinecone(index_name=self.index)

        vector = self.get_ada_embedding(observation)
        upsert_response = memory.upsert(
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
        return upsert_response

    def _fetch_memories(self, observation: str, namespace:str) -> List[Document]:
          #"""Fetch related characteristics, preferences or dislikes for a user."""
        query_embedding = self.get_ada_embedding(observation)
        memory = self.init_pinecone(index_name=self.index)
        memory.query(query_embedding, top_k=1, include_metadata=True, namespace=namespace,
                          filter={'user_id': {'$eq': self.user_id}})
    #     return self.memory_retriever.get_relevant_documents(observation)
    def _compute_agent_summary(self, model_speed:str):
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
        self.init_pinecone(index_name=self.index)
        # The agent seeks to think about their core characteristics.
        relevant_characteristics = self._fetch_memories(f"Users core characteristics", namespace="TRAITS")
        relevant_preferences = self._fetch_memories(f"Users core preferences", namespace="PREFERENCES")
        relevant_dislikes = self._fetch_memories(f"Users core dislikes", namespace="DISLIKES")
        if model_speed =='fast':
            output = self.replicate_llm(prompt)
            return output

        else:
            chain = LLMChain(llm=self.llm, prompt=prompt, verbose=self.verbose)
            return chain.run(name= self.user_id, relevant_characteristics=relevant_characteristics, relevant_preferences=relevant_preferences, relevant_dislikes=relevant_dislikes).strip()



    def update_agent_preferences(self, preferences:str):
        """Serves to update agents preferences so that they can be used in summary"""

        prompt = """ The {name} has following {past_preference} and the new {preferences}
                Update user preferences and return a list of preferences
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
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
        self.init_pinecone(index_name=self.index)
        past_dislikes = self._fetch_memories(f"Users core dislikes", namespace="DISLIKES")
        prompt = PromptTemplate(input_variables=["name", "past_dislikes", "dislikes"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_dislikes= past_dislikes, dislikes=dislikes)
        return self._update_memories(prompt, namespace="DISLIKES")


    def update_agent_traits(self, traits:str):
        """Serves to update agent traits so that they can be used in summary"""
        prompt =""" The {name} has following {past_traits} and the new {traits}
                Update user traits and return a list of traits
            Do not embellish.
            Summary: """
        self.init_pinecone(index_name=self.index)
        past_traits = self._fetch_memories(f"Users core dislikes", namespace="TRAITS")
        prompt = PromptTemplate(input_variables=["name", "past_traits", "traits"], template=prompt)
        prompt = prompt.format(name=self.user_id, past_traits= past_traits, traits=traits)
        return self._update_memories(prompt, namespace="TRAITS")


    def update_agent_summary(self):
        """Serves to update agent traits so that they can be used in summary"""
        summary = self._compute_agent_summary()
        return self._update_memories(summary, namespace="SUMMARY")

    def task_identification(self, goals:str):
        """Serves to update agent traits so that they can be used in summary"""
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        complete_query = str(agent_summary) + goals
        complete_query = PromptTemplate.from_template(complete_query)
        print("HERE IS THE COMPLETE QUERY", complete_query)
        from heuristic_experience_orchestrator.task_identification import TaskIdentificationChain
        chain = TaskIdentificationChain.from_llm(llm=self.llm, task_description="none",  value="Decomposition", verbose=self.verbose)

        chain_output = chain.run(name= self.user_id).strip()
        return chain_output


    def solution_generation(self, factors:dict, model_speed:str):
        """Generates a recipe solution in json"""
        import time

        start_time = time.time()
        prompt = """
                Help me choose what food choice, order, restaurant or a recipe to eat or make for my next meal.     
                There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                For '{{ factor }}', I want the meal to be '{{ value }}' points on a scale of 1 to 100 points{% if not loop.last %}.{% else %}.{% endif %}
                {% endfor %}
                Instructions and ingredients should be detailed.  Result type can be Recipe, but not Meal
                Answer with a result in a correct  python dictionary that is properly formatted that contains the following keys and must have  values
                "Result type",  "body" which should contain "title", "rating", "prep_time", "cook_time", "description", "ingredients", "instructions"
        """
        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        # complete_query =  output
        complete_query = PromptTemplate.from_template(complete_query)

        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            end_time = time.time()

            execution_time = end_time - start_time
            print("Execution time: ", execution_time, " seconds")
            return chain_result

    def goal_optimization(self, factors: dict, model_speed:str):
        """Serves to optimize agent goals"""

        prompt = """
              Based on all the history and information of this user, suggest three goals that are personal to him that he should apply to optimize his time.
              Only JSON values should be the output, don't write anything extra, no warnings no explanations. 
              Make sure to provide data in the following format
                         Answer with a result in a correct  python dictionary that is properly formatted that contains the following keys and must have  values     
              Goals containing 'body' that has multiple 'name', 'min', 'max', 'unit_name', 'option_array', 'name', 'min', 'max', 'unit_name', 'option_array', 'name', 'min', 'max', 'unit_name', 'option_array'
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        if model_speed =='fast':
            output = self.replicate_llm(output)
            return output
        else:
            chain = LLMChain(llm=self.llm, prompt=complete_query, verbose=self.verbose)
            chain_result = chain.run(prompt=complete_query, name=self.user_id).strip()
            return chain_result


    def restaurant_recommendation(self, factors: dict):
        """Serves to optimize agent goals"""

        prompt = """
              Based on the following factors, There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                Determine the type of restaurant you should offer to a customer. Make the reccomendation very short and to a point, as if it is something you would type on google maps
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        places = GooglePlacesTool()
        output = places.run(complete_query)
        return output

    def delivery_recommendation(self, factors: dict, model_speed:str):
        """Serves to optimize agent goals"""

        prompt = """
              Based on the following factors, There are {% for factor, value in factors.items() %}'{{ factor }}'{% if not loop.last %}, {% endif %}{% endfor %} factors I want to consider.
                {% for factor, value in factors.items() %}
                Determine the type of food you would want to recommend to the user, that is commonly ordered online. It should be like burger or pizza
            """

        self.init_pinecone(index_name=self.index)
        agent_summary = self._fetch_memories(f"Users core summary", namespace="SUMMARY")
        template = Template(prompt)
        output = template.render(factors=factors)
        complete_query = str(agent_summary) + output
        return wolt_tool.main(complete_query)

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
    agent.test_places()
    # agent._update_memories("lazy, stupid and hungry", "TRAITS")
    #agent.task_identification("I need your help choosing what to eat for my next meal. ")
    # agent.solution_generation( {    'health': 85,
    # 'time': 75,
    # 'cost': 50}, model_speed="slow")