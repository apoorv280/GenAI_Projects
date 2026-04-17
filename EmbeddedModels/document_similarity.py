from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

load_dotenv()  # Load environment variables from .env file

os.environ['HF_HOME'] = 'D:/huggingface_cache'  # Set the cache directory for Hugging Face models

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "Who is Sachin"

documents_vectors = embedding.embed_documents(documents)
query_vector = embedding.embed_query(query)

similarity_scores = cosine_similarity([query_vector], documents_vectors)[0]
sorted_scores = sorted(list(enumerate(similarity_scores)),key=lambda x:x[1], reverse=True)
for index, score in sorted_scores:
    print(f"Document: {documents[index]}, Similarity Score: {score:.4f}")