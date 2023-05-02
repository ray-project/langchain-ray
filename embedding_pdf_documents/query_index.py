"""
Query the local FAISS index.
Run this script only after running `embedding_ray.py`
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
query_embedding = HuggingFaceEmbeddings(model_name=model_name)
db = FAISS.load_local("faiss_index", query_embedding)
documents = db.similarity_search(query="prompt engineering", k=1)
print([doc.page_content for doc in documents])