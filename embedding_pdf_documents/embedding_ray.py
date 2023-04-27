import binascii
import io
from typing import List

import pypdf
import ray
from pypdf import PdfReader

ray.init(
    runtime_env={"pip": ["langchain", "pypdf", "sentence_transformers", "transformers"]}
)

ds = ray.data.read_binary_files("s3://arxiv-docs/")


def convert_to_text(pdf_bytes: bytes):
    pdf_bytes_io = io.BytesIO(pdf_bytes)

    try:
        pdf_doc = PdfReader(pdf_bytes_io)
    except pypdf.errors.PdfStreamError:
        # Skip pdfs that are not readable.
        # We still have over 30,000 pages after skipping these.
        return []

    text = []
    for page in pdf_doc.pages:
        try:
            text.append(page.extract_text())
        except binascii.Error:
            # Skip all pages that are not parseable due to malformed characters.
            print("parsing failed")
    return text
    

# We use `flat_map` as `convert_to_text` has a 1->N relationship.
# It produces N strings for each PDF (one string per page).
# Use `map` for 1->1 relationship.
ds = ds.flat_map(convert_to_text)

from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(page_text: str):
    # Use chunk_size of 1000.
    # We felt that the answer we would be looking for would be 
    # around 200 words, or around 1000 characters.
    # This parameter can be modified based on your documents and use case.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, length_function=len
    )
    split_text: List[str] = text_splitter.split_text(page_text)

    split_text = [text.replace("\n", " ") for text in split_text]
    return split_text


# We use `flat_map` as `split_text` has a 1->N relationship.
# It produces N output chunks for each input string.
# Use `map` for 1->1 relationship.
ds = ds.flat_map(split_text)

from sentence_transformers import SentenceTransformer

# Use LangChain's default model.
# This model can be changed depending on your task.
model_name = "sentence-transformers/all-mpnet-base-v2"


class Embed:
    def __init__(self):
        self.transformer = SentenceTransformer(model_name, device="cuda")

    def __call__(self, text_batch: List[str]):
        # We manually encode using sentence_transformer since LangChain
        # HuggingfaceEmbeddings does not support specifying a batch size yet.
        embeddings = self.transformer.encode(
            text_batch,
            batch_size=100,  # Large batch size to maximize GPU utilization.
            device="cuda",
        ).tolist()

        return list(zip(text_batch, embeddings))


# Use `map_batches` since we want to specify a batch size to maximize GPU utilization.
ds = ds.map_batches(
    Embed,
    # Large batch size to maximize GPU utilization.
    # Too large a batch size may result in GPU running out of memory.
    # If the chunk size is increased, then decrease batch size.
    # If the chunk size is decreased, then increase batch size.
    batch_size=100,  # Large batch size to maximize GPU utilization.
    compute=ray.data.ActorPoolStrategy(size=20),  # I have 20 GPUs in my cluster
    num_gpus=1,  # 1 GPU for each actor.
)

from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

text_and_embeddings = []
for output in ds.iter_rows():
    text_and_embeddings.append(output)

vectore_stor = FAISS.from_embeddings(
    text_and_embeddings,
    # Provide the embedding model to embed the query.
    # The documents are already embedded.
    embedding=HuggingFaceEmbeddings(model_name=model_name),
)

# Persist the vector store.
vectore_stor.save_local("faiss_index")
