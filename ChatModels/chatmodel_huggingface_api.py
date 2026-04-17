from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()


LLM = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation"
)
model = ChatHuggingFace(llm = LLM)

result = model.invoke("What is the capital of France")

print(result.content)

