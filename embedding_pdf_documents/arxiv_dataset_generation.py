"""The code that was used to generate the document dataset.

1. Make a search query to arxiv for the 2000 documents most relevant to "large language models"
2. For each result, download the PDF locally to a directory, with some error handling.
3. Upload the local directory to an AWS S3 bucket.
"""

import time
from urllib.error import HTTPError

import arxiv
from tqdm import tqdm

search_results = arxiv.Search(
    query="large language models",
    max_results=2000,
)

for result in tqdm(search_results):
    while True:
        try:
            result.download_pdf(dirpath="./arxiv_pdfs")
            break
        except FileNotFoundError:
            print("file not found")
            break
        except HTTPError:
            print("forbidden")
            break
        except ConnectionResetError as e:
            print("connection reset by peer")

            # wait for some time before retrying the connection
            time.sleep(5)

# Sync the local directory to S3
# aws s3 sync ./arxiv_pdfs s3://arxiv-docs/
