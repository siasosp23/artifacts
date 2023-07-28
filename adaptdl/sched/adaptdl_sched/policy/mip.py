# MIP formulation of Pollux with truncated search space for allocations
# Author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

import copy
import cvxpy as cp
import logging
import numpy as np
import time as time
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.cluster_config import ID_TO_NODENAME_MAP, NODENAME_TO_ID_MAP, CLUSTER_NUM_GPUS, BLACKLIST_NODES
from adaptdl_sched.cluster_config import alloc_to_placement_smart, get_mock_phoebe_node
from adaptdl.sched_hints import NODE_TO_CLUSTER_MAP

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

CONFIGS_4GPU = (np.asarray([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                np.asarray([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))

CONFIGS_8GPU = (np.asarray([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
                np.asarray([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]))

ZERO_ALLOC_GAIN = 0.01

DEBUG_PHOEBE = True

def get_restart_penalty(jobname):
    if "deep" in jobname:
        return 25
    elif "imagenet" in jobname:
        return 250
    elif "cifar" in jobname:
        return 20
    elif "bert" in jobname:
        return 120
    elif "yolov3" in jobname:
        return 80
    else:
        return 30

class MIPPolicy(object):
  # ensure sign(p_fairness) != sign(lambda_*)
  def __init__(self, 
         p_fairness=-0.5,
         lambda_a=0.01,
         lambda_n=1.1,
         project_throughputs=True,
         share_max_replicas=False,
         timeshare_penalty_window=None):
    # fairness param
    self.p_fairness = p_fairness

    # prev-allocs (config)
    self.prev_allocs = dict()
    # prev cluster (0 - slow, 1 - fast)
    self.prev_cluster = dict()

    # checkpoint-restart penalties (in seconds)
    self.restart_penalty = 30 # within-gpu-groups
    self.migrate_penalty = 30 # across-gpu-groups

    # possible configs to alloc from
    self.alloc_configs = None

    # optimization params
    self.lambda_a = lambda_a
    self.lambda_n = lambda_n

    # cluster ordering
    self.cluster_ordering = None

    # use max replicas across all clusters
    self.share_max_replicas = share_max_replicas

    # generalize throughputs to other clusters
    self.project_throughputs = project_throughputs
    self.gput_ratios = dict()
    if self.project_throughputs:
      LOG.info(f"Project speedups active")
      self.share_max_replicas = True
    if share_max_replicas:
      self.share_max_replicas = True

    # track allocations over a window
    self.apply_timeshare_penalty = timeshare_penalty_window is not None
    self.window_prev_allocs = dict()
    self.window_len = timeshare_penalty_window
      
  def get_valid_configs(self, nodes):
    # get gpu type -> nodenames map
    self.cluster_node_ordering = dict()
    for node_name, node_resources in nodes.items():
      if node_name in BLACKLIST_NODES:
          continue
      node_gpu_type = NODE_TO_CLUSTER_MAP.get(node_name, None)
      if node_gpu_type is None:
        LOG.error(f"Invalid node gpu type. Node = {node_gpu_type} -> {node_resources}")
      else:
        if node_gpu_type not in self.cluster_node_ordering:
          self.cluster_node_ordering[node_gpu_type] = []
        self.cluster_node_ordering[node_gpu_type].append(node_name)
    # get ordering between gpu types
    self.cluster_ordering = sorted(list(self.cluster_node_ordering.keys()))
    
    # node ordering
    for node_gpu_type in self.cluster_ordering:
      self.cluster_node_ordering[node_gpu_type] = sorted(self.cluster_node_ordering[node_gpu_type])

    self.cluster_num_nodes = {k : len(v) for k,v in self.cluster_node_ordering.items()}
    self.cluster_num_gpus = {k : CLUSTER_NUM_GPUS[k] for k in self.cluster_ordering}

    # get valid configs for each GPU type
    configs = dict()
    for node_gpu_type in self.cluster_ordering:
      configs[node_gpu_type] = []
      num_gpus_per_node = self.cluster_num_gpus[node_gpu_type]
      i = 1
      while i <= num_gpus_per_node:
        configs[node_gpu_type].append((1, i))
        i *= 2
      j = 2
      while j <= self.cluster_num_nodes[node_gpu_type]:
        configs[node_gpu_type].append((j, j * num_gpus_per_node))
        j += 1
    self.configs = configs
    
    new_nodes = dict()
    for cluster in self.cluster_ordering:
      cluster_node_dict = dict()
      for i, node_name in enumerate(self.cluster_node_ordering[cluster]):
        cluster_node_dict[i] = nodes[node_name]
      new_nodes[cluster] = cluster_node_dict
    return new_nodes, self.configs

  def _compute_goodputs(self, job_info, cluster_name, num_nodes, num_replicas):
    speedup_fn = job_info.speedup_fn.get(cluster_name, None)
    if speedup_fn is None and not self.project_throughputs:
      LOG.info(f"COMPUTE_GOODPUT:: {cluster_name}, {num_replicas} -> None")
      return None
    if speedup_fn is not None and isinstance(speedup_fn, SpeedupFunction):
      # speedup_fn exists for job in `cluster_name` cluster
      goodput_arr = np.asarray(speedup_fn.get_goodput(num_nodes.astype(np.float32), num_replicas.astype(np.float32)), dtype=np.float32)
      LOG.info(f"COMPUTE_GOODPUT:: {cluster_name}, {num_replicas} -> {goodput_arr}")
      return goodput_arr
    else:
      # assume linear scalability
      LOG.info(f"COMPUTE_GOODPUT:: {cluster_name}, {num_replicas} -> Linear")
      return num_replicas

    # self.project_throughputs and speedup_fn is None:
    # check if some speedup fn is not None
    any_speedup_fn = any([v is not None for v in job_info.speedup_fn.values()])
    if not any_speedup_fn:
      LOG.info(f"COMPUTE_GOODPUT:: {cluster_name}, {num_replicas} -> No speedup fns")
      return None
    # take any speedup_fn
    dest_cluster = [k for k in job_info.speedup_fn.keys() if job_info.speedup_fn[k] is not None][0]
    is_dest_cluster_4gpu = CLUSTER_NUM_GPUS[dest_cluster] == 4
    is_src_cluster_4gpu = CLUSTER_NUM_GPUS[cluster_name] == 4
    # both 8-GPU:
    if is_src_cluster_4gpu and is_dest_cluster_4gpu:
      translated_num_nodes, translated_num_replicas = num_nodes, num_replicas
    elif is_src_cluster_4gpu and not is_dest_cluster_4gpu:
      # 4 -> 8 GPU conversion
      translated_num_nodes = np.ceil(num_replicas / 8).astype(np.uint32)
      translated_num_replicas = num_replicas.astype(np.uint32)
    elif not is_src_cluster_4gpu and is_dest_cluster_4gpu:
      # 4 -> 8 GPU conversion
      translated_num_nodes = np.ceil(num_replicas / 4).astype(np.uint32)
      translated_num_replicas = num_replicas.astype(np.uint32)
    elif not is_src_cluster_4gpu and not is_dest_cluster_4gpu:
      translated_num_nodes, translated_num_replicas = num_nodes, num_replicas
    
    # remove configs that exceed cluster size
    max_dest_cluster_size = CLUSTER_NUM_GPUS[dest_cluster] * len(ID_TO_NODENAME_MAP[dest_cluster])
    valid_dest_configs_idxs = (translated_num_replicas <= max_dest_cluster_size) & (translated_num_nodes <= len(ID_TO_NODENAME_MAP[dest_cluster]))
    # multiplier for throughput projection
    multiplier = job_info.cluster_throughput_ratios[cluster_name][dest_cluster]

    # actual speedup_fn being evaluated
    dest_speedup_fn = job_info.speedup_fn[dest_cluster]

    # match output shape to input
    output_arr = np.zeros_like(num_nodes)
    translated_num_nodes = translated_num_nodes[valid_dest_configs_idxs]
    translated_num_replicas = translated_num_replicas[valid_dest_configs_idxs]

    goodput_arr = np.asarray(dest_speedup_fn.get_goodput(translated_num_nodes.astype(np.float32), translated_num_replicas.astype(np.float32)), dtype=np.float32)
    output_arr[valid_dest_configs_idxs] = goodput_arr * multiplier
    LOG.info(f"Imputated goodput for {cluster_name} from {dest_cluster}: {num_nodes}, {num_replicas} = {goodput_arr} -> {output_arr}")
    return output_arr
  
  def save_current_gput_ratios(self, cluster_matrices, clusters):
    gput_ratios = dict()
    for i, dst_cluster in enumerate(clusters):
      for j, src_cluster in enumerate(clusters):
        if dst_cluster == src_cluster:
          continue
        else:
          src_val = np.mean(cluster_matrices[src_cluster][:, 0])
          dst_val = np.mean(cluster_matrices[dst_cluster][:, 0])
          ratio = src_val / dst_val
          gput_ratios.setdefault(dst_cluster, dict())[src_cluster] = ratio
    self.gput_ratios = gput_ratios
  
  def get_current_gput_ratios(self):
    return self.gput_ratios if self.gput_ratios else dict()

  # currently broken.. will fix; focusing only on `optimize_mip_inv` since `p < 0` for eval
  def optimize_mip(self, jobs, nodes, prev_allocations):
    np.set_printoptions(suppress=True)
    joblist, jobnames = [], []
    for i, (jobname, job) in enumerate(jobs.items()):
      joblist.append(job)
      jobnames.append(jobname)

    # filter jobs to only contain active jobs first
    num_jobs = len(jobs)

    num_gpus = {}
    for k, k_nodes in nodes.items():
      num_gpus[k] = 0
      for node_idx, node in k_nodes.items():
        num_gpus[k] += node.resources.get("nvidia.com/gpu", 0)
    num_configs = {k : len(v[1]) for k, v in self.configs.items()}
    total_num_configs = sum(num_configs.values())

    # single-cluster speedup-matrix
    cluster_goodput_matrices = {k : np.zeros((num_jobs, num_configs[k]), dtype=np.float32) 
                  for k in num_configs.keys()
                  }
    realloc_factors, migrate_factors = [], []

    # job weights 
    job_weights = np.ones((1, num_jobs), dtype=np.float32)

    if self.apply_timeshare_penalty:
      timeshare_penalties = self.get_timeshare_penalties(jobs, jobnames)
    else:
      timeshare_penalties = None

    # compute raw speedup matrix (irrespective of slow/fast cluster)
    for i, job in enumerate(joblist):
      # compute _fair_ goodput
      for cluster in self.cluster_ordering:
        speedup_fn = job.speedup_fn[cluster]
        if self.share_max_replicas:
          max_replicas = max(job.max_replicas.values())
          min_replicas = min(job.min_replicas.values())
        else:
          max_replicas = job.max_replicas[cluster]
          min_replicas = job.min_replicas[cluster]
        if min_replicas > 1:
          LOG.info(f"Min replicas: {min_replicas} for job: {job.name}")
        
        if speedup_fn is None:
          if not self.project_throughputs:
            continue
          
          # check if any throughput model exists to extrapolate from
          any_speedup_fn = any([fn is not None for fn in job.speedup_fn.values()])
          if not any_speedup_fn:
            continue

        # cluster-specific configs
        alloc_num_nodes, alloc_num_replicas = self.configs[cluster]
        valid_configs = (alloc_num_replicas <= max_replicas) & (alloc_num_replicas >= min_replicas)
        valid_nnodes, valid_ngpus = alloc_num_nodes[valid_configs], alloc_num_replicas[valid_configs]
        goodput_matrix = cluster_goodput_matrices[cluster]
        valid_configs_goodput = self._compute_goodputs(job, cluster, valid_nnodes, valid_ngpus)
        # print(f"{jobnames[i]}, {cluster}: {valid_configs_goodput}")
        goodput_matrix[i, valid_configs] = valid_configs_goodput
      
      # fill in (1, 1) config for each cluster
      min_goodputs = []
      for cluster in self.cluster_ordering:
        if np.sum(cluster_goodput_matrices[cluster][i, :]) == 0:
          min_goodputs.append(0)
        else:
          min_goodputs.append(np.min(cluster_goodput_matrices[cluster][i, :][np.nonzero(cluster_goodput_matrices[cluster][i, :])]))
      min_goodputs = np.asarray(min_goodputs)
      if max(min_goodputs) == 0:
        min_goodput = 1
      else:
        # take lowest non-zero goodput value
        min_goodput = np.min(min_goodputs[np.nonzero(min_goodputs)])
      for cluster in self.cluster_ordering:
        # if goodput vector is empty, set first config to be [1]
        cluster_goodput_matrices[cluster][i, :] /= min_goodput
        if max(cluster_goodput_matrices[cluster][i, :]) == 0:
          cluster_goodput_matrices[cluster][i, 0] = 1.0
        # assert max(cluster_goodput_matrices[cluster][i, :]) < 500, "bad speedup values"
      
      # re-alloc/migrate factor
      job_restart_penalty = get_restart_penalty(jobnames[i])
      job_lost_gpu_seconds = (job.num_restarts * job_restart_penalty)
      realloc_factor = max((job.age - job_lost_gpu_seconds), 0) / (job.age + job_restart_penalty)
      migrate_factor = max((job.age - job_lost_gpu_seconds), 0) / (job.age + job_restart_penalty)
      realloc_factors.append(realloc_factor)
      migrate_factors.append(migrate_factor)
    
    self.save_current_gput_ratios(cluster_goodput_matrices, self.cluster_ordering)

    # append slow and fast speedup matrices
    final_speedup_matrix = np.hstack(tuple([cluster_goodput_matrices[cluster] for cluster in self.cluster_ordering]))

    # print(f"speedup matrix: {final_speedup_matrix}")
    optim_st_time = time.time()
    if self.apply_timeshare_penalty:
      job_allocs, cluster_allocs = self.__solve_mip_timeshare(final_speedup_matrix, num_gpus, job_weights,
                                  jobnames,realloc_factors, migrate_factors, 
                                  timeshare_penalties)
    else:
      job_allocs, cluster_allocs = self.__solve_mip_rescaled(final_speedup_matrix, num_gpus, job_weights, 
                                  jobnames, realloc_factors, migrate_factors)
    optim_ed_time = time.time()
    # TIME: {(optim_ed_time - optim_st_time) * 1000}ms")
    
    # cluster-specific job placements
    cluster_job_placements = dict()
    for cluster in self.cluster_ordering:
      node_remaining_gpus = np.asarray([node.resources.get("nvidia.com/gpu", 0) for idx, node in nodes[cluster].items()], dtype=np.uint32)
      if cluster in cluster_allocs:
        cluster_job_placements[cluster] = self.alloc_to_placement(cluster, cluster_allocs[cluster], node_remaining_gpus)
      else:
        cluster_job_placements[cluster] = dict()
    
    # merge allocs
    job_placements = {}
    for k, v in job_allocs.items():
      if v is None:
        job_placements[k] = (None, ())
      else:
        cluster_name, alloc = v
        job_placements[k] = (cluster_name, cluster_job_placements[cluster_name][k])

    return job_placements, 1

  def optimize_mip_inv(self, jobs, nodes, prev_allocations):
    np.set_printoptions(suppress=True)
    joblist, jobnames = [], []
    for i, (jobname, job) in enumerate(jobs.items()):
      joblist.append(job)
      jobnames.append(jobname)

    num_jobs = len(jobs)
    num_gpus = {}

    # filter jobs to only contain active jobs first
    for k, k_nodes in self.cluster_num_nodes.items():
      num_gpus[k] = k_nodes * self.cluster_num_gpus[k]
    num_configs = {k : len(v) for k, v in self.configs.items()}
    total_num_configs = sum(num_configs.values())
    LOG.info(f"Total number of configs per job: {total_num_configs}")

    # single-cluster speedup-matrix
    cluster_goodput_matrices = {k : np.zeros((num_jobs, num_configs[k]), dtype=np.float32) + ZERO_ALLOC_GAIN 
                  for k in num_configs.keys()
                  }
    realloc_factors, migrate_factors = [], []

    # job weights 
    job_weights = np.ones((1, num_jobs), dtype=np.float32)

    if self.apply_timeshare_penalty:
      timeshare_penalties = self.get_timeshare_penalties(jobs, jobnames)
    else:
      timeshare_penalties = None

    # compute raw speedup matrix (irrespective of slow/fast cluster)
    for i, job in enumerate(joblist):
      # compute _fair_ goodput
      nnz_speedups = dict()
      for cluster in self.cluster_ordering:
        speedup_fn = job.speedup_fn.get(cluster, None)
        if isinstance(job.max_replicas, dict):
          if self.share_max_replicas:
            max_replicas = max(job.max_replicas.values())
            min_replicas = min(job.min_replicas.values())
          else:
            max_replicas = job.max_replicas[cluster]
            min_replicas = job.min_replicas[cluster]
        else:
          max_replicas=  job.max_replicas
          min_replicas = job.min_replicas

        if min_replicas > 1:
          LOG.info(f"Min replicas: {min_replicas} for job: {jobnames[i]}")
        
        if speedup_fn is None:
          if not self.project_throughputs:
            nnz_speedups[cluster] = 1
            continue
          
          # check if any throughput model exists to extrapolate from
          any_speedup_fn = any([fn is not None for fn in job.speedup_fn.values()])
          if not any_speedup_fn:
            nnz_speedups[cluster] = 1
            continue

        # cluster-specific configs
        cluster_configs = self.configs[cluster]
        alloc_num_nodes, alloc_num_replicas = np.asarray([v[0] for v in cluster_configs]), np.asarray([v[1] for v in cluster_configs])
        valid_configs = (alloc_num_replicas <= max_replicas) & (alloc_num_replicas >= min_replicas)
        valid_nnodes, valid_ngpus = alloc_num_nodes[valid_configs], alloc_num_replicas[valid_configs]
        goodput_matrix = cluster_goodput_matrices[cluster]
        valid_configs_goodput = self._compute_goodputs(job, cluster, valid_nnodes, valid_ngpus)
        LOG.info(f"job={jobnames[i]}, gpu_type={cluster},  valid_nnodes={valid_nnodes}, valid_ngpus={valid_ngpus} -> {valid_configs_goodput}")
        if valid_configs_goodput is None or valid_nnodes.size == 0:
          nnz_speedups[cluster] = 1
        else:
          goodput_matrix[i, valid_configs] = valid_configs_goodput
      
      # fill in (1, 1) config for each cluster with lowest value for 1,1 config in any cluster
      cluster_min_goodputs = []
      for cluster in self.cluster_ordering:
        if cluster not in nnz_speedups:
          nnz_valid_idxs = cluster_goodput_matrices[cluster][i, :] > ZERO_ALLOC_GAIN
          cluster_min_goodputs.append(np.min(cluster_goodput_matrices[cluster][i, :][nnz_valid_idxs]))
      cluster_min_goodputs = np.asarray(cluster_min_goodputs)
      if cluster_min_goodputs.size == 0:
        min_goodput = 1
      else:
        # take lowest non-zero goodput value
        min_goodput = np.min(cluster_min_goodputs)

      for cluster in self.cluster_ordering:
        # normalize every config to min goodput
        cluster_goodput_matrices[cluster][i, :] /= min_goodput
        # if goodput vector is empty, set first config to be [1]
        if cluster in nnz_speedups:
          cluster_goodput_matrices[cluster][i, 0] = 1.0
        # assert max(cluster_goodput_matrices[cluster][i, :]) < 500, "bad speedup values"
      
      # re-alloc/migrate factor
      # TODO :: suhasj --> incorporate custom migration penalty
      job_lost_gpu_seconds = (job.num_restarts * self.restart_penalty)
      realloc_factor = max((job.age - job_lost_gpu_seconds), 0) / (job.age + self.restart_penalty)
      realloc_factors.append(realloc_factor)
      migrate_factors.append(1)
    
    self.save_current_gput_ratios(cluster_goodput_matrices, self.cluster_ordering)

    # append slow and fast speedup matrices
    final_speedup_matrix = np.hstack(tuple([cluster_goodput_matrices[cluster] for cluster in self.cluster_ordering]))

    LOG.info(f"Speedup Matrix: {final_speedup_matrix}")
    optim_st_time = time.time()
    if self.apply_timeshare_penalty:
      job_allocs, cluster_allocs = self.__solve_mip_timeshare(final_speedup_matrix, num_gpus, job_weights,
                                  jobnames,realloc_factors, migrate_factors, 
                                  timeshare_penalties)
    else:
      job_allocs, cluster_allocs = self.__solve_mip_rescaled(final_speedup_matrix, num_gpus, job_weights, 
                                  jobnames, realloc_factors, migrate_factors)
    optim_ed_time = time.time()
    # TIME: {(optim_ed_time - optim_st_time) * 1000}ms")
    
    # cluster-specific job placements
    cluster_job_placements = dict()
    for cluster in self.cluster_ordering:
      node_remaining_gpus = np.asarray([node.resources.get("nvidia.com/gpu", 0) for idx, node in nodes[cluster].items()], dtype=np.uint32)
      if cluster in cluster_allocs:
        cluster_job_placements[cluster] = alloc_to_placement_smart(cluster, cluster_allocs[cluster], prev_allocations, node_remaining_gpus)
      else:
        cluster_job_placements[cluster] = dict()
    
    # merge allocs and rename each GPU ID to full k8s node names
    job_placements = {}
    for k, v in job_allocs.items():
      if v is None:
        job_placements[k] = []
      else:
        cluster_name, _ = v
        placement = cluster_job_placements[cluster_name][k]
        new_placement = []
        for gpu_id in placement:
          node_name = ID_TO_NODENAME_MAP[cluster_name][gpu_id]
          new_placement.append(node_name)
        job_placements[k] = new_placement
    
    # log placements to stdout
    LOG.info(f"Placements: {job_placements}")
    return job_placements, 1

  # alternate formulation with speedups scaled for reallocation
  # cluster_num_gpus = (num_slow_gpus, num_fast_gpus)
  def __solve_mip_rescaled(self, speedup_matrix, num_gpus, job_weights = None, jobnames=None, 
               realloc_factors=None, migrate_factors=None, timeshare_penalties=None):
    # ones vec
    num_jobs, num_configs = speedup_matrix.shape
    cluster_config_offset, cluster_num_configs = {}, {}
    idx = 0
    for cluster in self.cluster_ordering:
      cluster_config_offset[cluster] = idx
      cluster_num_configs[cluster] = len(self.configs[cluster])
      idx += cluster_num_configs[cluster]
    ones_jobvec = np.ones((1, num_jobs), dtype=np.float32)
    ones_configvec = np.ones((num_configs, 1), dtype=np.float32)

    # slow/fast config selectors (one-hot vectors)
    # cluster selectors
    cluster_selectors = dict()
    cluster_selectors["none"] = np.ones((num_configs), dtype=np.bool8)
    for cluster in self.cluster_ordering:
      cluster_config_selector = np.zeros((num_configs), dtype=np.bool8)
      low, high = cluster_config_offset[cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
      cluster_config_selector[low : high] = True
      cluster_selectors[cluster] = cluster_config_selector

    # job weights
    if job_weights is None:
      job_weights = ones_jobvec

    # rescale speedups for differing allocations
    for i, jobname in enumerate(jobnames):
      prev_alloc = self.prev_allocs.get(jobname, None)
      prev_cluster = self.prev_cluster.get(jobname, None)
      same_cluster_selector = cluster_selectors[prev_cluster] if prev_cluster else cluster_selectors["none"]
      migrate_cluster_selector = ~same_cluster_selector

      # rescale speedups projecting reallocations / migrations
      if prev_cluster is not None:
        same_cluster_realloc = same_cluster_selector & ~prev_alloc
        migrate_cluster_realloc = migrate_cluster_selector & ~prev_alloc

        # no penalty for keeping same alloc; penalty for realloc within same cluster
        speedup_matrix[i, :] = np.where(same_cluster_realloc, realloc_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])
      
        # rescale speedups projecting migrations
        speedup_matrix[i, :] = np.where(migrate_cluster_realloc, migrate_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

    # power-up speedup matrix 
    A = np.power(speedup_matrix, self.p_fairness)

    A = np.round(A, decimals=2)
    # TODO(suhasj): fix this clipping of matrix
    CLIP_FACTOR = 2 / ZERO_ALLOC_GAIN
    A[A > CLIP_FACTOR] = CLIP_FACTOR
    # print(f"OPTIM: Jobnames: {jobnames}")
    # print(f"OPTIM: Input matrix: {A}")

    # regularization parameter
    opt_lambda_alloc_change = self.lambda_a if self.lambda_a else -0.02
    opt_lambda_no_alloc = self.lambda_n if self.lambda_n else -1
    constraints = []

    # construct variable to optimize over
    x = cp.Variable(shape=A.shape, integer=True)
    
    # construct objective : weighted sum of mul(x, A) with weights
    obj_expr = cp.sum(job_weights @ cp.multiply(x, A))
    if jobnames is not None:
      for i, jobname in enumerate(jobnames):
        # penalty for no-alloc in both sub-clusters
        obj_expr += opt_lambda_no_alloc * (1 - cp.sum(x[i, :]))

        # no previous allocation
        if jobname not in self.prev_allocs:
          continue
        else:
          prev_alloc = self.prev_allocs[jobname]

        # penalty for change of allocation
        t_job = cp.Variable(shape=A[i, :].shape)
        obj_expr += opt_lambda_alloc_change * cp.sum(t_job)
        constraints.append(t_job >= (prev_alloc - x[i, :]))
        constraints.append(t_job >= (x[i, :] - prev_alloc))
        constraints.append(t_job <= 1)

    if self.p_fairness < 0:
      obj = cp.Minimize(obj_expr)
    else:
      obj = cp.Maximize(obj_expr)
    
    # add constraints
    # constrain range of x
    constraints.append(x >= 0)
    constraints.append(x <= 1)

    # constrain max-number of gpus alloc'ed per sub-cluster
    # slow sub-cluster
    for cluster in self.cluster_ordering:
      start_offset, end_offset = cluster_config_offset[cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
      alloc_configs = self.configs[cluster]
      config_ngpus = np.asarray([v[1] for v in alloc_configs], dtype=np.uint32)
      constraints.append(cp.sum(x[:, start_offset : end_offset] @ config_ngpus) <= num_gpus[cluster])

    # constraint only one config per job
    constraints.append((x @ ones_configvec) <= ones_jobvec)

    problem = cp.Problem(obj, constraints=constraints)
    st_time = time.time()
    problem.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF'}, verbose=False)
    ed_time = time.time()
    # print(f"OPTIM: Problem: {problem}")

    if problem.status != 'optimal':
      LOG.error(f"Solver time: {ed_time - st_time}s")
      LOG.error("Status: ", problem.status)
      LOG.error(f"Problem: {problem}")
      LOG.error("The optimal value is", problem.value)
      LOG.error("A solution x is")
      LOG.error(np.round(x.value))
      LOG.error("Using solution anyway.. hoping things change next round..")
    
    # record new allocs as prev-alloc for next iter
    output_soln = np.round(x.value, decimals=0).astype(np.uint32)

    # convert binary solution to allocations
    job_allocs, cluster_allocs = dict(), dict()
    cluster_config_offset_list = np.asarray([cluster_config_offset[cluster] for cluster in self.cluster_ordering])
    for i, jobname in enumerate(jobnames):
      soln = output_soln[i, :]
      if np.sum(soln) == 0:
        job_allocs[jobname] = None
        cluster_name = None
      else:
        # map solution to allocation
        alloc_config_idx = np.nonzero(soln)[0][0]
        cluster_id = np.nonzero(alloc_config_idx >= cluster_config_offset_list)[0][-1]
        cluster_name = self.cluster_ordering[cluster_id]
        config_idx = alloc_config_idx - cluster_config_offset_list[cluster_id]
        nnodes, ngpus = self.configs[cluster_name][config_idx][0], self.configs[cluster_name][config_idx][1]
        job_allocs[jobname] = (cluster_name, (nnodes, ngpus))
        if cluster_name not in cluster_allocs:
          cluster_allocs[cluster_name] = dict()
        cluster_allocs[cluster_name][jobname]= (nnodes, ngpus)
        
      self.prev_allocs[jobnames[i]] = soln
      self.prev_cluster[jobnames[i]] = cluster_name
    return job_allocs, cluster_allocs
  
  # improved version that attempts to rotate resources between jobs when num jobs > num resources
  def __solve_mip_timeshare(self, speedup_matrix, num_gpus, job_weights = None, jobnames=None, 
               realloc_factors=None, migrate_factors=None, timeshare_penalties=None):
    # ones vec
    num_jobs, num_configs = speedup_matrix.shape
    cluster_config_offset, cluster_num_configs = {}, {}
    idx = 0
    for cluster in self.cluster_ordering:
      cluster_config_offset[cluster] = idx
      cluster_num_configs[cluster] = len(self.configs[cluster][0])
      idx += cluster_num_configs[cluster]
    ones_jobvec = np.ones((1, num_jobs), dtype=np.float32)
    ones_configvec = np.ones((num_configs, 1), dtype=np.float32)

    # slow/fast config selectors (one-hot vectors)
    # cluster selectors
    cluster_selectors = dict()
    cluster_selectors["none"] = np.ones((num_configs), dtype=np.bool8)
    for cluster in self.cluster_ordering:
      cluster_config_selector = np.zeros((num_configs), dtype=np.bool8)
      low, high = cluster_config_offset[cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
      cluster_config_selector[low : high] = True
      cluster_selectors[cluster] = cluster_config_selector

    # job weights
    if job_weights is None:
      job_weights = ones_jobvec

    # rescale speedups for differing allocations
    for i, jobname in enumerate(jobnames):
      prev_alloc = self.prev_allocs.get(jobname, None)
      prev_cluster = self.prev_cluster.get(jobname, None)
      same_cluster_selector = cluster_selectors[prev_cluster] if prev_cluster else cluster_selectors["none"]
      migrate_cluster_selector = ~same_cluster_selector

      # rescale speedups projecting reallocations / migrations
      if prev_cluster is not None:
        same_cluster_realloc = same_cluster_selector & ~prev_alloc
        migrate_cluster_realloc = migrate_cluster_selector & ~prev_alloc

        # no penalty for keeping same alloc; penalty for realloc within same cluster
        speedup_matrix[i, :] = np.where(same_cluster_realloc, realloc_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])
      
        # rescale speedups projecting migrations
        speedup_matrix[i, :] = np.where(migrate_cluster_realloc, migrate_factors[i] * speedup_matrix[i, :], speedup_matrix[i, :])

    # power-up speedup matrix 
    A = np.power(speedup_matrix, self.p_fairness)

    A = np.round(A, decimals=2)
    # print(f"OPTIM: Jobnames: {jobnames}")
    # print(f"OPTIM: Input matrix: {A}")

    # regularization parameter
    opt_lambda_alloc_change = self.lambda_a if self.lambda_a else -0.02
    opt_lambda_no_alloc = self.lambda_n if self.lambda_n else -1
    constraints = []

    # construct variable to optimize over
    x = cp.Variable(shape=A.shape, integer=True)

    if timeshare_penalties is not None:
      penalty_no_alloc, penalty_change_alloc, penalty_no_change = timeshare_penalties

    # construct objective : weighted sum of mul(x, A) with weights
    obj_expr = cp.sum(job_weights @ cp.multiply(x, A))
    if jobnames is not None:
      for i, jobname in enumerate(jobnames):
        # penalty for no-alloc in both sub-clusters
        no_alloc_gain = -1 * penalty_no_alloc[i] * (1 - cp.sum(x[i, :]))
        obj_expr += no_alloc_gain

        # no previous allocation
        if jobname not in self.prev_allocs:
          continue
        else:
          prev_alloc = self.prev_allocs[jobname]

        # gain of service
        # job_window_gain_vec = np.where(prev_alloc, penalty_no_change[i], penalty_change_alloc[i])
        # job_change_gain = (1 - cp.sum(cp.multiply(x[i, :], job_window_gain_vec)))
        # obj_expr += job_change_gain

    obj = cp.Maximize(obj_expr)
    
    # add constraints
    # constrain range of x
    constraints.append(x >= 0)
    constraints.append(x <= 1)

    # constrain max-number of gpus alloc'ed per sub-cluster
    # slow sub-cluster
    for cluster in self.cluster_ordering:
      start_offset, end_offset = cluster_config_offset[cluster], cluster_config_offset[cluster] + cluster_num_configs[cluster]
      config_nnodes, config_ngpus = self.configs[cluster]
      constraints.append(cp.sum(x[:, start_offset : end_offset] @ config_ngpus) <= num_gpus[cluster])

    # constraint only one config per job
    constraints.append((x @ ones_configvec) <= ones_jobvec)

    problem = cp.Problem(obj, constraints=constraints)
    st_time = time.time()
    problem.solve(solver=cp.GLPK_MI, verbose=False)
    ed_time = time.time()
    # print(f"OPTIM: Problem: {problem}")

    if problem.status != 'optimal':
      print(f"Solver time: {ed_time - st_time}s")
      print("Status: ", problem.status)
      print(f"Problem: {problem}")
      print("The optimal value is", problem.value)
      print("A solution x is")
      print(np.round(x.value))
    
    # record new allocs as prev-alloc for next iter
    output_soln = np.round(x.value, decimals=0).astype(np.uint32)

    # update penalties for next iteration
    self.update_timeshare_penalties({jobnames[i] : output_soln[i, :] for i in range(len(jobnames))})

    # convert binary solution to allocations
    job_allocs, cluster_allocs = dict(), dict()
    cluster_config_offset_list = np.asarray([cluster_config_offset[cluster] for cluster in self.cluster_ordering])
    effective_gpus = dict()
    for i, jobname in enumerate(jobnames):
      soln = output_soln[i, :]
      if np.sum(soln) == 0:
        job_allocs[jobname] = None
        cluster_name = None
        effective_gpus[jobname] = 0
      else:
        # map solution to allocation
        alloc_config_idx = np.nonzero(soln)[0][0]
        cluster_id = np.nonzero(alloc_config_idx >= cluster_config_offset_list)[0][-1]
        cluster_name = self.cluster_ordering[cluster_id]
        config_idx = alloc_config_idx - cluster_config_offset_list[cluster_id]
        nnodes, ngpus = self.configs[cluster_name][0][config_idx], self.configs[cluster_name][1][config_idx]
        job_allocs[jobname] = (cluster_name, (nnodes, ngpus))
        if cluster_name not in cluster_allocs:
          cluster_allocs[cluster_name] = dict()
        cluster_allocs[cluster_name][jobname]= (nnodes, ngpus)
        effective_gpus[jobname] = speedup_matrix[i, alloc_config_idx]

      self.prev_allocs[jobnames[i]] = soln
      self.prev_cluster[jobnames[i]] = cluster_name
    # print(f"Effective GPUs: {sum(effective_gpus.values())}")
    return job_allocs, cluster_allocs

  def optimize(self, jobs, nodes, base_allocations, node_template):
    LOG.info(f"Input nodes: {nodes}")
    LOG.info(f"Input jobs: {jobs}")
    LOG.info(f"Input base_allocations: {base_allocations}")
    # TODO :: jobs[i].speedup_fn is not a map : gpu_type -> gpu_speedup_fn
    if DEBUG_PHOEBE:
      # blacklist all other gpu types except `chosen_cluster`
      add_nodes = ["phodgx1", "phodgx2", "phoquad1", "phortx1", "phortx2", "phortx3"]
      new_nodes = { k : get_mock_phoebe_node(k, nodes['phoebe-mgmt']) for k in add_nodes}
      nodes = new_nodes
      LOG.info(f"DEBUG_PHOEBE: Input nodes: {nodes}")

      for jobname in jobs.keys():
        # no speedup fn if 
        if not isinstance(jobs[jobname].speedup_fn, dict):
            jobs[jobname].speedup_fn = dict()
        
    new_nodes, alloc_configs = self.get_valid_configs(nodes)
    LOG.info(f"Post-filtering nodes: {new_nodes}")
    '''
    # get size of clusters
    cluster_num_nodes, cluster_num_gpus = dict(), dict()
    for cluster, cluster_nodes in new_nodes.items():
      cluster_num_nodes[cluster] = len(new_nodes[cluster])
      cluster_num_gpus[cluster] = 0
      for idx, node_info in cluster_nodes.items():
        cluster_num_gpus[cluster] += node_info.resources.get('nvidia.com/gpu', 0)
    LOG.info(f"Optimize: cluster_num_nodes: {cluster_num_nodes}, cluster_num_gpus: {cluster_num_gpus}")
    '''
    LOG.info(f"Alloc configs: {alloc_configs}")
    LOG.info(f"Fairness knob: p = {self.p_fairness}")
    if self.p_fairness > 0:
      return self.optimize_mip(jobs, new_nodes, base_allocations)
    elif self.p_fairness < 0:
      return self.optimize_mip_inv(jobs, new_nodes, base_allocations)
    else:
      LOG.error(f"Invalid p value : {self.p_fairness}")
      return None
