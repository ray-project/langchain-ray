import os
import time
from typing import Any, List, Optional

import ray
import torch
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.llms.utils import enforce_stop_tokens
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from ray import serve
from starlette.requests import Request
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)
from transformers import pipeline as hf_pipeline
from wandb.integration.langchain import WandbTracer

from local_embeddings import LocalHuggingFaceEmbeddings


class StableLMPipeline(HuggingFacePipeline):
    """A StableLM Pipeline that executes its workload locally.

    It monkey patches two methods.
    - _call to allow for the correct passing in of stop tokens.
    - from_model_id to allow for using the appropriate torch.dtype to use
      float16.

    This class is temporary, we are working with the authors of LangChain to make these
    unnecessary.
    """

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.pipeline(
            prompt, temperature=0.1, max_new_tokens=256, do_sample=True
        )
        if self.pipeline.task == "text-generation":
            # Text generation return includes the starter text.
            print(f"Response is: {response}")
            text = response[0]["generated_text"][len(prompt) :]
        else:
            raise ValueError(f"Got invalid task {self.pipeline.task}. ")
        # text = enforce_stop_tokens(text, [50278, 50279, 50277, 1, 0])
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
