# pip install arxiv, tqdm

import arxiv
import time
from tqdm import tqdm
from urllib.error import HTTPError

search_results = arxiv.Search(
  query= "large language models",
  max_results = 2000,
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