"""This file is only intended for benchmarking purposes. 
Use `embedding_ray.py` fir the actual LangChain+Ray code.
"""

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

# Put your list of file paths here
pdf_documents = [...]

langchain_documents = []
for document in pdf_documents:
    loader = PyPDFLoader(document)
    data = loader.load()
    langchain_documents.append(data)

split_docs = text_splitter.split_documents(langchain_documents)
db = FAISS.from_documents(split_docs, embedding=hf)
