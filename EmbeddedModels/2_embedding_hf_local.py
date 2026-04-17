from langchain_huggingface import HuggingFaceEmbeddings

import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set the cache directory for Hugging Face models

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India"

# vector = embedding.embed_query(text)
# print(str(vector))

documents = [
    "Paris is the capital of France",
    "Berlin is the capital of Germany", 
    "Madrid is the capital of Spain"
]

vectors = embedding.embed_documents(documents)
print(str(vectors))
