# Teenage-AGI

## Objective
Inspired by the several Auto-GPT related Projects (predominently BabyAGI) and the Paper ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442), this python project uses OpenAI and Pinecone to Give memory to an AI agent and also allows it to "think" before making an action (outputting text). Also, just by shutting down the AI, it doesn't forget its memories since it lives on Pinecone and its memory_counter saves the index that its on.

### Sections
- [How it Works](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#how-it-works)
- [How to Use](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#how-to-use)
- [Experiments](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#experiments)
- [More About Project & Me](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#how-to-use)
- [Credits](https://github.com/seanpixel/Teenage-AGI/blob/main/README.md#credits)

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
4. `open .env` and set your OpenAI and Pinecone API info.
5. Run `python main.py` and talk to the AI in the terminal

## Running in a docker container
You can run the system isolated in a container using docker-compose:
```
docker-compose run teenage-agi
```

## Experiments
Currently, using GPT-4, I found that it can remember its name and other characteristics. It also carries on the conversation quite well without a context window (although I might add it soon). I will update this section as I keep playing with it.

## More about the Project & Me
After reading the Simulcra paper, I made this project in my college dorm. I realized that most of the "language" that I generate are inside my head, so I thought maybe it would make sense if AGI does as well. I'm a founder currently runing a startup called [DSNR]([url](https://www.dsnr.ai/)) and also a first-year at USC. Contact me on [twitter](https://twitter.com/sean_pixel) about anything would love to chat.

## Credits
Thank you to [@yoheinakajima](https://twitter.com/yoheinakajima) and the team behind ["Generative Agents: Interactive Simulacra of Human Behavior"](https://arxiv.org/abs/2304.03442) for the idea!
