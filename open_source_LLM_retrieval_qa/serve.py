import ray
import os
from starlette.requests import Request
from ray import serve
from typing import List, Optional, Any
from langchain.llms.utils import enforce_stop_tokens

from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
from wandb.integration.langchain import WandbTracer
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
import time
from local_embeddings import LocalHuggingFaceEmbeddings
from local_pipelines import StableLMPipeline

FAISS_INDEX_PATH = "faiss_index_fast"


template = """
<|SYSTEM|># StableLM Tuned (Alpha version)
- You are a helpful, polite, fact-based agent for answering questions about Ray. 
- Your answers include enough detail for someone to follow through on your suggestions. 
<|USER|>
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Please answer the following question using the context provided. 

CONTEXT: 
{context}
=========
QUESTION: {question} 
ANSWER: <|ASSISTANT|>"""
PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])


@serve.deployment(ray_actor_options={"num_gpus": 1})
class QADeployment:
    def __init__(self):
        WandbTracer.init({"project": "retrieval_demo"})

        # Load the data from faiss. No change from Part 1
        st = time.time()
        self.embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
        self.db = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings)
        et = time.time() - st

        print(f"Loading FAISS database took {et} seconds.")
        st = time.time()

        self.llm = StableLMPipeline.from_model_id(
            model_id="stabilityai/stablelm-tuned-alpha-7b",
            task="text-generation",
            model_kwargs={"device_map": "auto", "torch_dtype": torch.float16},
        )
        et = time.time() - st
        print(f"Loading HF model took {et} seconds.")
        self.chain = load_qa_chain(llm=self.llm, chain_type="stuff", prompt=PROMPT)

    def qa(self, query):
        search_results = self.db.similarity_search(query)
        print(f"Results from db are: {search_results}")
        result = self.chain({"input_documents": search_results, "question": query})

        print(f"Result is: {result}")
        return result["output_text"]

    async def __call__(self, request: Request) -> List[str]:
        return self.qa(request.query_params["query"])


deployment = QADeployment.bind()
