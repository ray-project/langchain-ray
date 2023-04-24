import time

import numpy as np
import ray
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from embeddings import LocalHuggingFaceEmbeddings

# To download the files locally for processing, here's the command line
# wget -e robots=off --recursive --no-clobber --page-requisites --html-extension \
# --convert-links --restrict-file-names=windows \
# --domains docs.ray.io --no-parent https://docs.ray.io/en/master/

FAISS_INDEX_PATH = "faiss_index_fast"
db_shards = 8

loader = ReadTheDocsLoader("docs.ray.io/en/master/")

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
)


@ray.remote(num_gpus=1)
def process_shard(shard):
    print(f"Starting process_shard of {len(shard)} chunks.")
    st = time.time()
    embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
    result = FAISS.from_documents(shard, embeddings)
    et = time.time() - st
    print(f"Shard completed in {et} seconds.")
    return result


# Stage one: read all the docs, split them into chunks.
st = time.time()
print("Loading documents ...")
docs = loader.load()
# Theoretically, we could use Ray to accelerate this, but it's fast enough as is.
chunks = text_splitter.create_documents(
    [doc.page_content for doc in docs], metadatas=[doc.metadata for doc in docs]
)
et = time.time() - st
print(f"Time taken: {et} seconds. {len(chunks)} chunks generated")

# Stage two: embed the docs.
print(f"Loading chunks into vector store ... using {db_shards} shards")
st = time.time()
shards = np.array_split(chunks, db_shards)
futures = [process_shard.remote(shards[i]) for i in range(db_shards)]
results = ray.get(futures)
et = time.time() - st
print(f"Shard processing complete. Time taken: {et} seconds.")

st = time.time()
print("Merging shards ...")
# Straight serial merge of others into results[0]
db = results[0]
for i in range(1, db_shards):
    db.merge_from(results[i])
et = time.time() - st
print(f"Merged in {et} seconds.")

st = time.time()
print("Saving faiss index")
db.save_local(FAISS_INDEX_PATH)
et = time.time() - st
print(f"Saved in: {et} seconds.")
