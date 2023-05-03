# Scaling Embedding Generation with LangChain and Ray

This is a more detailed walkthrough covering how to create the multi-node Ray cluster and run the code for our blog post. 

The full Ray cluster launcher documentation can be found [here](https://docs.ray.io/en/latest/cluster/getting-started.html).

## Step 1
Install Ray locally: `pip install 'ray[default]'`

## Step 2
Clone the repository `git clone https://github.com/ray-project/langchain-ray/` and switch into the directory
`cd langchain-ray`.
You can edit the [cluster yaml file](llm-batch-inference.yaml) if you need to make changes.

## Step 3
Setup the necessary AWS credentials (set the `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `AWS_SESSION_TOKEN` environment variables).
Then, you can start a Ray cluster via this YAML file: `ray up -y llm-batch-inference.yaml`

## Step 4
You can connect via the remote Ray dashboard: `ray dashboard llm-batch-inference.yaml`. 
This will setup the necessary port forwarding.

The dashboard can be viewed by visiting `http://localhost:8265`

## Step 5
You can view the progress of the worker node startup by viewing the autoscaler status on the Ray dashboard

![Screen Shot 2023-04-24 at 10 28 44 PM](https://user-images.githubusercontent.com/8068268/234182585-66ab4778-8a4b-4c34-acee-a0671ecd2fa7.png)

## Step 6
Copy the [requirements.txt](requirements.txt) file and the [Ray batch inference code](embedding_ray.py) to the Ray cluster:

```
ray rsync_up llm-batch-inference.yaml 'embedding_ray.py' 'embedding_ray.py'
ray rsync_up llm-batch-inference.yaml 'requirements.txt' 'requirements.txt'
```

## Step 7
In a separate window, SSH into the Ray cluster via `ray attach llm-batch-inference.yaml`

## Step 8
Install the requirements on the head node of the cluster

`pip install -r requirements.txt`

## Step 9
Once all the worker nodes have started, run the [Ray batch inference code](embedding_ray.py) on the cluster!

`python embedding_ray.py`

## Step 10

After the workload finished, tear down the cluster! This needs to be run from your laptop, so if you are
still in the cluster shell, make sure to exit with Control-C.

`ray down llm-batch-inference.yaml`