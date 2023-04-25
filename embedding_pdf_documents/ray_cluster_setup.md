# Creating a Ray cluster with 20 GPUs

This is a more detailed walkthrough covering how to create the multi-node Ray cluster for our blog post. 
The full Ray cluster launcher documentation can be found [here](https://docs.ray.io/en/latest/cluster/getting-started.html).

## Step 1
Download the [following cluster yaml file](llm-batch-inference.yaml) locally:

```yaml
# An unique identifier for the head node and workers of this cluster.
cluster_name: llm-batch-inference

max_workers: 5

# Cloud-provider specific configuration.
provider:
    type: aws
    region: us-west-2
    # In case you need to specify a pre-generated EC2 SSH key.
    # In this case, you need to download the key from the EC2 console, 
    # move it to your ~/.ssh/ directory and grant the correct permissions like `chmod 400 ~/.ssh/llm-blogpost-test-key.pem`
    # Then, uncomment the below lines and replace `key_name` with the name of your key:
    # key_pair:
    #   key_name: llm-blogpost-test-key

available_node_types:
  ray.head.default:
      resources: {"CPU": 48, "GPU": 4}
      node_config:
        InstanceType: g4dn.12xlarge
        BlockDeviceMappings:
            - DeviceName: /dev/sda1
              Ebs:
                  VolumeSize: 200
  ray.worker.default:
      node_config:
        InstanceType: g4dn.12xlarge
        BlockDeviceMappings:
            - DeviceName: /dev/sda1
              Ebs:
                  VolumeSize: 200
      resources: {"CPU": 48, "GPU": 4}
      min_workers: 4
      max_workers: 4
```

## Step 2

Then, you can start a Ray cluster via this YAML file: `ray up -y llm-batch-inference.yaml`

## Step 3

You can connect via the remote Ray dashboard: `ray dashboard llm-batch-inference.yaml`. 
This will set up the necessary port forwarding.

The dashboard can be viewed by visiting `http://localhost:8265`

## Step 4

You can view the progress of the worker node startup by viewing the autoscaler status on the Ray dashboard

![Screen Shot 2023-04-24 at 10 28 44 PM](https://user-images.githubusercontent.com/8068268/234182585-66ab4778-8a4b-4c34-acee-a0671ecd2fa7.png)

## Step 5

Copy the [requirements.txt](requirements.txt) file and the [Ray batch inference code](embedding_ray.py) to the Ray cluster:

```
ray rsync_up llm-batch-inference.yaml 'embedding_ray.py' 'embedding_ray.py'
ray rasync_up llm-batch-inference.yaml 'requirements.txt' 'requirements.txt'
```

## Step 6

In a separate window, SSH into the Ray cluster via `ray attach llm-batch-inference.yaml`

## Step 7

Install the requirements on the head node of the cluster

`pip install -r requirements.txt`

## Step 8

Run the [Ray batch inference code](embedding_ray.py) on the cluster!

`python embedding_ray.py`
