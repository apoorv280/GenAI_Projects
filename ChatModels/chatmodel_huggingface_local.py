from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from torch.cuda import temperature
from transformers import pipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set the cache directory for Hugging Face models

pipe = pipeline(
    "text-generation",
    model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    max_new_tokens = 1000,
    temperature = 0.7
)

LLM = HuggingFacePipeline(pipeline=pipe)

model = ChatHuggingFace(llm = LLM)
result = model.invoke("What is the capital of USA?")
print(result.content)