import ray 
import os
from starlette.requests import Request
from ray import serve
from typing import List, Optional, Any
from langchain.llms.utils import enforce_stop_tokens
from local_embeddings import LocalHuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline as hf_pipeline
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from wandb.integration.langchain import WandbTracer

import torch

import time

FAISS_INDEX_PATH = 'faiss_index_fast' 


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


class StableLMPipeline(HuggingFacePipeline):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        
        response = self.pipeline(prompt, temperature=0.1, max_new_tokens=256, do_sample=True)
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            print(f'Response is: {response}')
            text = response[0]["generated_text"][len(prompt) :]
        else:
            raise ValueError(
                f"Got invalid task {self.pipeline.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            # This is a bit hacky, but I can't figure out a better way to enforce
            # stop tokens when making calls to huggingface_hub.
            text = enforce_stop_tokens(text, [50278, 50279, 50277, 1, 0])
        return text
    
    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        task: str,
        device: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        **kwargs: Any,
    ):
        """Construct the pipeline object from model_id and task."""

        pipeline = hf_pipeline(
            model=model_id, 
            task=task,
            device=device,
            model_kwargs=model_kwargs,
        )
        return cls(
            pipeline=pipeline,
            model_id=model_id,
            model_kwargs=model_kwargs,
            **kwargs,
        )

@serve.deployment(ray_actor_options={"num_gpus":1})
class QADeployment:
    def __init__(self):
        WandbTracer.init({"project": "wandb_prompts_2"})
        #Load the data from faiss
        st = time.time()
        self.embeddings = LocalHuggingFaceEmbeddings('multi-qa-mpnet-base-dot-v1')
        self.db = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings)
        et = time.time() - st
        print(f'Loading FAISS database took {et} seconds.')
        st = time.time() 
        self.llm = StableLMPipeline.from_model_id(model_id="stabilityai/stablelm-tuned-alpha-7b", 
                                                     task="text-generation", model_kwargs=
                                                     {"device_map":"auto", "torch_dtype": torch.float16})
        et = time.time() - st
        print(f'Loading HF model took {et} seconds.')
        self.chain = load_qa_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=PROMPT)
    

    def search(self,query): 
        results = self.db.max_marginal_relevance_search(query)
        retval = ''
        for i in range(len(results)):
            chunk = results[i]
            source = chunk.metadata['source']
            retval = retval + f'From http://{source}\n\n'
            retval = retval + chunk.page_content
            retval = retval + '\n====\n\n'
                           
        return retval
    
    def qa(self, query):
        search_results = self.db.similarity_search(query)
        print(f'Results from db are: {search_results}')
        result = self.chain({"input_documents": search_results, "question":query})
        
        print(f'Result is: {result}')
        return result["output_text"]
    
    async def __call__(self, request: Request) -> List[str]:
        return self.qa(request.query_params["query"])

deployment = QADeployment.bind()
