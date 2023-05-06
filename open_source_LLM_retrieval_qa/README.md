# Building a retrival-based Question Answering system using LangChain + Ray in 20 minutes

This is source code for "Building a retrival-based Question Answering system using LangChain + Ray in 20 minutes"

## Requirements 

This demo requires a bit of a hefty setup. It requires one machine with a 24GB GPU (eg. an AWS g5.xlarge) or a machine with 2 GPUs 
(minimum 16GB each) or a Ray cluster with at least 2 GPUs available. 

Torch drivers are not well optimized for the M2 yet.  This demo will run on an M2 Mac, but: 

- It will take almost 30 minutes to build the index instead of about 2 minutes with a GPU. 
- You need to remove the GPU requirements (num_gpus=0) that are in the @serve.deployment (serve.py) and 
the @ray.remote() declaration of build_vector_store.py, as well as the float16 in the serve.py file 
- You better go get a cup of coffee. A query that takes 5 seconds with a GPU takes 3-4 minutes with the CPU. 

This demo uses Weights and Biases for prompt observability. If you do not want to 

## Getting started

Once you have a cloud machine with the required specifications, do `pip install ray[default]`

To apply the indexing to the Ray docs, use `tar xvfz docs.ray.io.tar.gz`

## Installing dependencies

To install required dependencies, do `pip install -r requirements.txt` 

## Building the vector store index

To build the index do `python build_vector_store.py`

## Serving

To serve do `serve run serve:deployment` 

This sets up a service on port 8000 by default. 

## Querying

To query, we've included a simple python script. 

`python query.py 'What is the difference between SERVE and PACK placement groups?'`

