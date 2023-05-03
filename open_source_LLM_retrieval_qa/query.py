import sys

import requests

query = sys.argv[1]
response = requests.post(f"http://localhost:8000/?query={query}")
print(response.content.decode())
