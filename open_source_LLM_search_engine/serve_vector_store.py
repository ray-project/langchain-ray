import time
from typing import List

from langchain.vectorstores import FAISS
from ray import serve
from starlette.requests import Request

from embeddings import LocalHuggingFaceEmbeddings

FAISS_INDEX_PATH = "faiss_index"


@serve.deployment
class VectorSearchDeployment:
    def __init__(self):
        # Load the data from faiss
        st = time.time()
        self.embeddings = LocalHuggingFaceEmbeddings("multi-qa-mpnet-base-dot-v1")
        self.db = FAISS.load_local(FAISS_INDEX_PATH, self.embeddings)
        et = time.time() - st
        print(f"Loading database took {et} seconds.")

    def search(self, query):
        results = self.db.max_marginal_relevance_search(query)
        retval = ""
        for i in range(len(results)):
            chunk = results[i]
            source = chunk.metadata["source"]
            retval = retval + f"From http://{source}\n\n"
            retval = retval + chunk.page_content
            retval = retval + "\n====\n\n"

        return retval

    async def __call__(self, request: Request) -> List[str]:
        return self.search(request.query_params["query"])


deployment = VectorSearchDeployment.bind()
