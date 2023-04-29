import time
import subprocess
import openai



def openai_call(self,
                prompt: str,
                model: str = None,
                temperature: float = None,
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