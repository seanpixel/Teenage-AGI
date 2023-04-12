import openai
import os
import pinecone
import yaml
from dotenv import load_dotenv

# Load default environment variables (.env)
load_dotenv()


def generate(prompt):
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
        {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
        {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"]

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
#PINECONE_API_ENV = "asia-southeast1-gcp"
    
# Prompt Initialization
with open('prompts.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# Counter Initialization
with open('memory_count.yaml', 'r') as f:
    counter = yaml.load(f, Loader=yaml.FullLoader)

# internalThoughtPrompt = data['internal_thought']
# externalThoughtPrompt = data['external_thought']
# internalMemoryPrompt = data['internal_thought_memory']
# externalMemoryPrompt = data['external_thought_memory']

THOUGHTS = "Thoughts"
k_n = 5

# initialize pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

# initialize openAI
openai.api_key = OPENAI_API_KEY # you can just copy and paste your key here if you want

def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]


class Agent():
    def __init__(self, table_name=None) -> None:
        self.table_name = table_name
        self.memory = None
        self.thought_id_count = int(counter['count'])

    # Keep Remebering!
    # def __del__(self) -> None:
    #     with open('memory_count.yaml', 'w') as f:
    #         yaml.dump({'count': str(self.thought_id_count)}, f)
    

    def createIndex(self, table_name=None):
        # Create Pinecone index
        if(table_name):
            self.table_name = table_name

        if(self.table_name == None):
            return

        dimension = 1536
        metric = "cosine"
        pod_type = "p1"
        if self.table_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
            )

        # Give memory
        self.memory = pinecone.Index(self.table_name)

    
    # Adds new "Thought" to agent. thought_type is Query, Internal, and External
    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        vector = get_ada_embedding(new_thought)
        upsert_response = self.memory.upsert(
        vectors=[
            {
            'id':f"thought-{self.thought_id_count}", 
            'values':vector, 
            'metadata':
                {"thought_string": new_thought, 
                 "thought_type": thought_type}
            }],
	    namespace=THOUGHTS,
        )

        self.thought_id_count += 1

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query) -> str:
        query_embedding = get_ada_embedding(query)
        results = self.memory.query(query_embedding, top_k=k_n, include_metadata=True, namespace=THOUGHTS)
        sorted_results = sorted(results.matches, key=lambda x: x.score, reverse=True)
        top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
        #print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches)
        # print("------------INTERNAL THOUGHT PROMPT------------")
        # print(internalThoughtPrompt)
        internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        #print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought)
        self.updateMemory(internalMemoryPrompt, "Internal")
        return internal_thought, top_matches

    def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)
        
        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought)
        # print("------------EXTERNAL THOUGHT PROMPT------------")
        # print(externalThoughtPrompt)
        external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, "External")
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), "Query")
        return external_thought
    
    # Make agent read some information (learn) WIP
    def read(self, text) -> str:
        pass





    



    
