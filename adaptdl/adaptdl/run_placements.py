#!/usr/bin/env python3

import argparse
import copy
import itertools
import os
import subprocess
import time
import yaml
from kubernetes import client, config, watch

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="name of model")
parser.add_argument("repository", type=str, help="url to docker repository")
args = parser.parse_args()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(project_root, "benchmark", "models", args.model)

with open(os.path.join(model_dir, "adaptdljob.yaml")) as f:
    template = yaml.load(f)

dockerfile = os.path.join(model_dir, "Dockerfile")
image = args.repository + ":" + args.model
subprocess.check_call(["docker", "build", "-t", image, project_root, "-f", dockerfile])
subprocess.check_call(["docker", "tag", image, "docker.pdl.cmu.edu/" + image])
image = "docker.pdl.cmu.edu/" + image
subprocess.check_call(["docker", "push", image])
repodigest = subprocess.check_output(
        ["docker", "image", "inspect", image, "--format={{index .RepoDigests 0}}"])
repodigest = repodigest.decode().strip()

template["spec"]["template"]["spec"]["containers"][0]["image"] = repodigest

config.load_kube_config()
core_api = client.CoreV1Api()
objs_api = client.CustomObjectsApi()
namespace = config.list_kube_config_contexts()[1]["context"].get("namespace", "default")
obj_args = ("adaptdl.petuum.com", "v1", namespace, "adaptdljobs")

node_list = core_api.list_node(label_selector="!node-role.kubernetes.io/master")
nodes = [n.metadata.name for n in node_list.items]
print("Nodes:", nodes)
gpus = [int(n.status.capacity.get("nvidia.com/gpu", "0")) for n in node_list.items]
nodes, gpus = nodes[2], gpus[2]
nodes = ["phodgx1", "phodgx2"]
gpus = [8, 8]
print("Chosen nodes:", nodes)
print("Chosen #GPUs:", gpus)

def normalize(p):
    p = tuple(filter(None, p))
    return min(p[i:] + p[:i] for i in range(len(p)))

placements = itertools.product(*map(range, [g + 1 for g in gpus]))
placements = sorted(set(normalize(p) for p in placements if sum(p) > 0))
print(f"Placements: {placements}")

for i, p in enumerate(placements):
    print("Job {}/{}: {}".format(i + 1, len(placements), p))
    job = copy.deepcopy(template)
    name = job["metadata"].pop("generateName") + "".join(map(str, p))
    job["metadata"]["name"] = name
    job["spec"]["placement"] = []
    for n, k in zip(nodes, p):
        job["spec"]["placement"].extend([n] * k)
    volumes = job["spec"]["template"]["spec"].setdefault("volumes", [])
    volumes.append({
        "name": "pollux",
        "persistentVolumeClaim": { "claimName": "pollux" },
    })
    mounts = job["spec"]["template"]["spec"]["containers"][0].setdefault("volumeMounts", [])
    mounts.append({
        "name": "pollux",
        "mountPath": "/pollux",
    })
    mounts.append({
        "name": "pollux",
        "mountPath": "/pollux/checkpoint",
        "subPath": "pollux/checkpoint/" + name,
    })
    mounts.append({
        "name": "pollux",
        "mountPath": "/pollux/tensorboard",
        "subPath": "pollux/tensorboard/" + name,
    })
    env = job["spec"]["template"]["spec"]["containers"][0].setdefault("env", [])
    env.append({"name": "ADAPTDL_CHECKPOINT_PATH", "value": "/pollux/checkpoint"})
    env.append({"name": "ADAPTDL_TENSORBOARD_LOGDIR", "value": "/pollux/tensorboard"})
    env.append({"name": "TRACE_THROUGHPUT", "value": "true"})
    print(yaml.dump(job))
    objs_api.create_namespaced_custom_object(*obj_args, job)
    while True:
        # Wait for job to be completed.
        obj = objs_api.get_namespaced_custom_object(*obj_args, name)
        if obj.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            break
        time.sleep(5)
    while True:
        # Wait for pods to be completed.
        pod_list = core_api.list_namespaced_pod(namespace, label_selector="adaptdl/job={}".format(name))
        if all(pod.status.phase in ("Succeeded", "Failed") for pod in pod_list.items):
            break
        time.sleep(5)
