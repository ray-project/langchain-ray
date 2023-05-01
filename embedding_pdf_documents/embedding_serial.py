"""This file is only intended for benchmarking purposes. 
Use `embedding_ray.py` for the actual LangChain+Ray code.
"""
import os
from tqdm import tqdm

from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    length_function=len,
)
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Put your directory containing PDFs here
directory = '/tmp/data/'
pdf_documents = [os.path.join(directory, filename) for filename in os.listdir(directory)]

langchain_documents = []
for document in tqdm(pdf_documents):
    try:
        loader = PyPDFLoader(document)
        data = loader.load()
        langchain_documents.extend(data)
    except Exception:
        continue

print("Num pages: ", len(langchain_documents))
print("Splitting all documents")
split_docs = text_splitter.split_documents(langchain_documents)

print("Embed and create vector index")
db = FAISS.from_documents(split_docs, embedding=hf)
