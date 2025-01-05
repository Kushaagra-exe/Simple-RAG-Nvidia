from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


api = os.getenv('NVIDIA_API')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api
)

completion = client.chat.completions.create(
  model="meta/llama-3.1-405b-instruct",
  messages=[{"role":"user","content":"What are Artificial Neural Networks"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

