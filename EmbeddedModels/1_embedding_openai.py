from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

result = embedding.embed_query("Delhi is the capital of India")

print(str(result))


embedding = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=32)

documents = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany", 
    "Madrid is the capital of Spain"
]


result = embedding.embed_documents(documents)

print(str(result))



