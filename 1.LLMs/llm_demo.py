from langchain_openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model = 'gpt-3.5-turbo-instruct')

result = llm.invoke('what is the capital of France?')
print(result)