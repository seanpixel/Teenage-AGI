import openai
import os
import chromadb
import yaml
from dotenv import load_dotenv
import nltk
from langchain.text_splitter import NLTKTextSplitter



# Download NLTK for Reading
nltk.download('punkt')

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)

# Load default environment variables (.env)
load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4"

def generate(prompt):
    completion = openai.ChatCompletion.create(
    model=OPENAI_MODEL,
    messages=[
        {"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
        {"role": "system", "content": "Keep your thoughts relatively simple and concise"},
        {"role": "user", "content": prompt},
        ]
    )

    return completion.choices[0].message["content"]


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    
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

# Thought types, used in Pinecone Namespace
THOUGHTS = "Thoughts"
QUERIES = "Queries"
INFORMATION = "Information"
ACTIONS = "Actions"

# Top matches length
k_n = 3


# initialize openAI
openai.api_key = OPENAI_API_KEY # you can just copy and paste your key here if you want

def get_ada_embedding(text):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]


class Agent():
    def __init__(self, name) -> None:
        chroma_client = chromadb.Client()
        self.memory = chroma_client.create_collection(name)
        self.thought_id_count = int(counter['count'])
        self.last_message = ""
        self.memory.add(
                    documents=["You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."],
                    metadatas=[{"thought_type" : THOUGHTS}],
                    ids=[str(self.thought_id_count)]
                )
        
    
    # Adds new Memory to agent, types are: THOUGHTS, ACTIONS, QUERIES, INFORMATION
    def updateMemory(self, new_thought, thought_type):
        with open('memory_count.yaml', 'w') as f:
             yaml.dump({'count': str(self.thought_id_count)}, f)

        if thought_type==INFORMATION:
            new_thought = "This is information fed to you by the user:\n" + new_thought
        elif thought_type==QUERIES:
            new_thought = "The user has said to you before:\n" + new_thought
        elif thought_type==THOUGHTS:
            # Not needed since already in prompts.yaml
            # new_thought = "You have previously thought:\n" + new_thought
            pass
        elif thought_type==ACTIONS:
            # Not needed since already in prompts.yaml as external thought memory
            pass

        self.memory.add(
            documents=[new_thought],
            metadatas=[{"thought_type": thought_type}],
            ids=[str(self.thought_id_count)]
        )

        self.thought_id_count += 1

    # Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
    def internalThought(self, query) -> str:
        n = k_n
        if k_n > self.memory.count():
            n = self.memory.count()

        results = []

        try:
            query_results = self.memory.query(
                query_texts=[query],
                n_results=n,
                where={"thought_type": QUERIES}
            )
            results += query_results["documents"][0]
        except:
            pass

        if n > self.memory.count():
            n = self.memory.count()

        try:
            thought_results = self.memory.query(
                query_texts=[query],
                n_results=n,
                where={"thought_type": THOUGHTS}
            )
            results += thought_results["documents"][0]
        except:
            pass

        top_matches = "\n\n".join(results) # WIP
        #print(top_matches)
        
        internalThoughtPrompt = data['internal_thought']
        internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{last_message}", self.last_message)
        print("------------INTERNAL THOUGHT PROMPT------------")
        print(internalThoughtPrompt)
        internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
        
        # Debugging purposes
        #print(internal_thought)

        internalMemoryPrompt = data['internal_thought_memory']
        internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        self.updateMemory(internalMemoryPrompt, THOUGHTS)
        return internal_thought, top_matches

    def action(self, query) -> str:
        internal_thought, top_matches = self.internalThought(query)
        
        externalThoughtPrompt = data['external_thought']
        externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
        print("------------EXTERNAL THOUGHT PROMPT------------")
        print(externalThoughtPrompt)
        external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

        externalMemoryPrompt = data['external_thought_memory']
        externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
        self.updateMemory(externalMemoryPrompt, THOUGHTS)
        request_memory = data["request_memory"]
        self.updateMemory(request_memory.replace("{query}", query), QUERIES)
        self.last_message = query
        return external_thought

    # Make agent think some information
    def think(self, text) -> str:
        self.updateMemory(text, THOUGHTS)


    # Make agent read some information
    def read(self, text) -> str:
        texts = text_splitter.split_text(text)

        metadatas = []
        counted_ids = []

        for i in range(0, len(texts)):
            texts[i] = "This is information fed to you by the user:\n" + texts[i]
            metadatas.append({"thought_type", INFORMATION})
            counted_ids.append(str(self.thought_id_count))
            self.thought_id_count += 1

        self.memory.add(
            texts,
            metadatas=metadatas,
            ids=counted_ids
        )





   
