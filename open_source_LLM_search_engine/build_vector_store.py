import time

from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from embeddings import LocalHuggingFaceEmbeddings

# To download the files locally for processing, here's the command line
# wget -e robots=off --recursive --no-clobber --page-requisites --html-extension \
# --convert-links --restrict-file-names=windows \
# --domains docs.ray.io --no-parent https://docs.ray.io/en/master/

FAISS_INDEX_PATH = "faiss_index"

loader = ReadTheDocsLoader("docs.ray.io/en/master/")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)

# Stage one: read all the docs, split them into chunks.
st = time.time()
print("Loading documents ...")
docs = loader.load()
# Theoretically, we could use Ray to accelerate this, but it's fast enough as is.
chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
)
et = time.time() - st
print(f"Time taken: {et} seconds.")

# Stage two: embed the docs.
embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
print(f"Loading chunks into vector store ...")
st = time.time()
db = FAISS.from_documents(chunks, embeddings)
db.save_local(FAISS_INDEX_PATH)
et = time.time() - st
print(f"Time taken: {et} seconds.")
