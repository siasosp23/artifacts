import copy
import cvxpy as cp
import logging
import numpy as np
import time as time
from adaptdl.sched_hints import NODE_TO_CLUSTER_MAP
from collections import Counter

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

ID_TO_NODENAME_MAP = {
  "dgx" : {0 :"phodgx1", 1 : "phodgx2"},
  "rtx" : {0 : "phortx1", 1 : "phortx2", 2 : "phortx3"},
  "quad" : {0 : "phoquad1"}
}

NODENAME_TO_ID_MAP = {
        "dgx" : {"phodgx1" : 0, "phodgx2": 1},
        "rtx" : {"phortx1" : 0, "phortx2" : 1, "phortx3" : 2},
        "quad" : {"phoquad1" : 0}
}

CLUSTER_NUM_GPUS = {
  "dgx" : 8,
  "rtx" : 8,
  "quad" : 4,
}

# do not consider these nodes for scheduling
BLACKLIST_NODES = ["phoebe-mgmt"]

def get_gavel_cluster_config(active_nodes):
  cluster_num_nodes = {}
  cluster_ngpus_per_node = {}
  for node_name in active_nodes:
    gpu_type = NODE_TO_CLUSTER_MAP[node_name]
    if gpu_type not in cluster_num_nodes:
      cluster_num_nodes[gpu_type] = 0
      cluster_ngpus_per_node[gpu_type] = CLUSTER_NUM_GPUS[gpu_type]
  
    cluster_num_nodes[gpu_type] += 1
  return cluster_num_nodes, cluster_ngpus_per_node

# returns a mock node for phoebe cluster
def get_mock_phoebe_node(node_name, template_node):
  gpu_type = NODE_TO_CLUSTER_MAP[node_name]
  ngpus_per_node = CLUSTER_NUM_GPUS[gpu_type]
  copy_node = copy.deepcopy(template_node)
  copy_node.resources['nvidia.com/gpu'] = ngpus_per_node
  copy_node.resources['cpu'] = 12800000
  copy_node.resources['memory'] = 60393771318325
  copy_node.resources['pods'] = 128
  return copy_node

## Attempts to realize allocations in `job_allocs` using nodes in `node_remaining_gpus`
## while respecting current placements in `cur_placement` for `cluster_name` gpu type
# Args: cluster_name: cluster name/gpu type
#       job_allocs: {jobname : (num_nodes, num_gpus)}
#       cur_placements: {jobname : (cluster, [gpu0, gpu1, gpu2, ...])}
#       node_remaining_gpus: [gpus_left_in_node_0, gpus_left_in_node_1, .. N-1]
def alloc_to_placement_smart(cluster_name, job_allocs, cur_placements, node_remaining_gpus):
  LOG.info(f"Cluster: {cluster_name}")
  LOG.info(f"Allocs: {job_allocs}")
  LOG.info(f"Cur Placements: {cur_placements}")
  max_num_nodes = len(node_remaining_gpus)

  # convert node names to node IDs for prev allocs
  prev_placements = dict()
  for jobname, placement in cur_placements.items():
    if len(placement) > 0:
      old_cluster_name = NODE_TO_CLUSTER_MAP[placement[0]]
      # migrated between GPU types
      if old_cluster_name != cluster_name:
        prev_placements[jobname] = [-1]*len(placement)
      else:
        prev_placement = [NODENAME_TO_ID_MAP[cluster_name][node_name] for node_name in placement]
        prev_placements[jobname] = prev_placement
    else:
      prev_placements[jobname] = []
  
  # determined {jobname : [gpu0, gpu1, gpu2...]}
  placed_jobs = dict()
  # partition into distributed and single-node jobs
  single_node_jobs, distributed_jobs = [], []
  ngpus_per_node = CLUSTER_NUM_GPUS[cluster_name]
  for jobname, (nnodes, ngpus) in job_allocs.items():
    if ngpus >= ngpus_per_node:
      distributed_jobs.append(jobname)
    else:
      single_node_jobs.append(jobname)
  # preserve placements for no change in alloc
  distr_placed_jobs = dict()
  single_placed_jobs = dict()
  for jobname in job_allocs.keys():
    prev_gpus = prev_placements.get(jobname, [])
    _, cur_ngpus = job_allocs.get(jobname, (0, 0))
    prev_cluster = None
    # valid allocation in this cluster
    if len(prev_gpus) > 0 and prev_gpus[0] >= 0:
      prev_cluster = cluster_name
    if prev_cluster == cluster_name and len(prev_gpus) == cur_ngpus:
      if cur_ngpus < ngpus_per_node:
        single_placed_jobs[jobname] = prev_gpus
      else:
        distr_placed_jobs[jobname] = prev_gpus
      for node_id in prev_gpus:
        node_remaining_gpus[node_id] -= 1
        # print(f"Preserving placement: {jobname} -> {cluster_name}, {prev_gpus}")
  
  # alloc any other distr jobs from last node ID
  for jobname in distributed_jobs:
    # skip if preserving placement
    if jobname in distr_placed_jobs:
      continue
    nnodes, ngpus = job_allocs.get(jobname, (0, 0))
    assert ngpus != 0, f"got zero gpus for {jobname}"
    # allocate nodes from last node ID
    cur_node_id = max_num_nodes - 1
    job_placement = []
    while nnodes > 0 and cur_node_id >= 0:
      # take whole node
      if node_remaining_gpus[cur_node_id] == ngpus_per_node:
        job_placement.extend([int(cur_node_id)] * ngpus_per_node)
        node_remaining_gpus[cur_node_id] = 0
        nnodes -= 1
      cur_node_id -= 1
      if cur_node_id == -1 and nnodes > 0:
        # reclaim some node from single-node jobs
        reclaim_node_id = np.argmax(node_remaining_gpus)
        # print(f"reclaiming node --> {reclaim_node_id}, remaining gpus = {node_remaining_gpus[reclaim_node_id]}")
        # find jobs mapped to this node
        reclaim_jobs = []
        for reclaim_jobname in single_placed_jobs.keys():
          if reclaim_node_id in single_placed_jobs[reclaim_jobname]:
            reclaim_jobs.append(reclaim_jobname)
        for reclaim_jobname in reclaim_jobs:
          gpus = single_placed_jobs.pop(reclaim_jobname)
          # print(f"evicting {reclaim_jobname} -> {gpus}")
          for node_id in gpus:
            node_remaining_gpus[node_id] += 1
        assert node_remaining_gpus[reclaim_node_id] == ngpus_per_node, "eviction assert"
        # loop again to find this freed machine
        cur_node_id = max_num_nodes - 1
    # ensure all nodes got placed
    assert nnodes == 0, f"couldnt place -- {jobname} -> {job_allocs[jobname]}"
    distr_placed_jobs[jobname] = job_placement
  # print(f"Distributed placements: {distr_placed_jobs}")
  
  # alloc any single node jobs from first node ID
  def get_job_order(joblist):
    return sorted(joblist, key=lambda x : job_allocs.get(jobname, (0, 0))[1], reverse=True)
  joblist = [jobname for jobname in single_node_jobs if jobname not in single_placed_jobs]
  # priority queue with prio = ngpus
  job_order = get_job_order(joblist)
  while len(job_order) > 0:
    jobname = job_order.pop(0)
    nnodes, ngpus = job_allocs.get(jobname, (0, 0))
    if ngpus == 0:
      single_placed_jobs[jobname] = None
      continue

    # allocate nodes from first node ID
    # prefer packing --> seek node id with min(ngpus) remaining after alloc
    job_placement = []
    idxs = np.arange(max_num_nodes)
    filter = node_remaining_gpus >= ngpus
    if not any(filter):
      # reclaim some gpus by evicting fewest gpus
      reclaim_cand_idxs = idxs[node_remaining_gpus < ngpus]
      reclaim_ordering = sorted(reclaim_cand_idxs, key=lambda x: (ngpus - node_remaining_gpus[x]))
      reclaim_node_id = reclaim_ordering[0]
      # evict some jobs from this node
      # print(f"reclaiming node --> {reclaim_node_id}")
      # find jobs mapped to this node
      reclaim_jobs = []
      for reclaim_jobname in single_placed_jobs.keys():
        if reclaim_node_id in single_placed_jobs[reclaim_jobname]:
          reclaim_jobs.append(reclaim_jobname)
      # sort from smallest to largest job in node
      reclaim_jobs = sorted(reclaim_jobs, key=lambda x : job_allocs[x][1])
      while node_remaining_gpus[reclaim_node_id] < ngpus and len(reclaim_jobs) > 0:
        reclaim_jobname = reclaim_jobs.pop(0)
        gpus = single_placed_jobs.pop(reclaim_jobname)
        # print(f"evicting {reclaim_jobname} -> {gpus}")
        for gpu_id in gpus:
          node_remaining_gpus[gpu_id] += 1
        # add back to placement queue
        job_order.append(reclaim_jobname)
      # update placement queue with priorities
      job_order = get_job_order(job_order)
      # print(f"new job order -> {job_order}")
      assert node_remaining_gpus[reclaim_node_id] >= ngpus, "eviction assert"
      filter = node_remaining_gpus >= ngpus
    assert any(filter), "failed to find a node to place: {jobname}; allocs = {job_allocs}, prev_allocs = {prev_allocs}, node_remaining_gpus = {node_remaining_gpus}"
    # simple packing algo -- most full valid placement
    place_idxs = idxs[filter]
    place_idxs = sorted(place_idxs, key=lambda x: node_remaining_gpus[x])
    place_idx = place_idxs[0]
    job_placement.extend([int(place_idx)] * ngpus)
    node_remaining_gpus[place_idx] -= ngpus
    single_placed_jobs[jobname] = job_placement
  # print(f"Single node placements: {single_placed_jobs}")
  placed_jobs = distr_placed_jobs
  placed_jobs.update(single_placed_jobs)
  
  return placed_jobs

# fixes allocations by keeping GPUs making up majority of allocations (dropping others)
# prio_order: ordering of GPU types to keep if equal probability of dropping
# returns: new_allocs --> fixed allocs
#          drop_count --> number of GPUs dropped for each job
def fix_allocations_by_dropping(allocs, prio_order=None):
  if prio_order is None:
    prio_order = {"dgx" : 0, "quad": 1, "rtx": 2}
  new_allocs = dict()
  drop_count = dict()
  for k, v in allocs.items():
    # count unique GPUs of each type
    gpu_types = [NODE_TO_CLUSTER_MAP[v2] for v2 in v]
    gpu_counts = Counter(gpu_types).most_common()

    # no heterogeneous allocs
    if len(set(gpu_counts)) == 1:
      new_allocs[k] = v
      drop_count[k] = 0
      continue
    
    if v is None or len(v) == 0:
      new_allocs[k] = v
      drop_count[k] = 0
      continue

    # homogeneous allocs
    chosen_gpus = gpu_counts[0]
    if gpu_counts[0][1] == gpu_counts[1][1]:
      # check priority; keep more powerful GPUs in tie-break
      chosen_prio = prio_order[gpu_counts[0][0]]
      next_prio = prio_order[gpu_counts[1][0]]
      if next_prio < chosen_prio:
        chosen_gpus = gpu_counts[1]
    
    # retain gpus of chosen type
    chosen_type = chosen_gpus[0]
    new_allocs[k] = []
    for v2 in v:
      if NODE_TO_CLUSTER_MAP[v2] == chosen_type:
        new_allocs[k].append(v2)
      else:
        LOG.info(f"Dropping: {v2} for {k}")
  return new_allocs, drop_count
