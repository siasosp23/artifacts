# Implements the Sia scheduling algorithm using a MIP solver
# Mathematically equivalent to Pollux policy for a homogeneous cluster
# Author: Suhas Jayaram Subramanya (suhasjs@cs.cmu.edu)

import cvxpy as cp
import numpy as np
import time as time
from copy import deepcopy

import simulator_config as sim_config
from speedup import *

CONFIGS_4GPU = (np.asarray([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                np.asarray([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))

CONFIGS_8GPU = (np.asarray([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
                np.asarray([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]))

ZERO_ALLOC_GAIN = 0.0001

# restart penalties for each job measured on real physical cluster
# jobnames are typically of the form '<model>-<jobnumber>', so this is okay
def get_restart_penalty(jobname):
    if "deepspeech" in jobname:
        return 25
    elif "imagenet" in jobname:
        return 250
    elif "cifar" in jobname:
        return 30
    elif "bert" in jobname:
        return 120
    elif "yolov3" in jobname:
        return 45
    else:
        return 30


class SiaPolicy(object):
    def __init__(self, p_fairness=sim_config.sia_default_p_val, restart_penalty=30,
                 migrate_penalty=45, lambda_a=sim_config.sia_default_lambda_a, 
                 lambda_n=sim_config.sia_default_lambda_n, project_throughputs=False,
                 share_max_replicas=False, sched_interval=60, enable_bsz_tuning=True):
        self.p_fairness = p_fairness

        # prev-allocs (config)
        self.prev_allocs = dict()
        # prev cluster (0 - slow, 1 - fast)
        self.prev_cluster = dict()

        # penalties for restart / migration
        self.restart_penalty = restart_penalty
        self.migrate_penalty = migrate_penalty

        # possible configs to alloc from
        self.alloc_configs = None

        # optimization params
        self.lambda_a = lambda_a
        self.lambda_n = lambda_n
        print(
            f"Solver params: p={self.p_fairness}, lambda_a={self.lambda_a}, lambda_n={self.lambda_n}")

        # cluster ordering
        self.cluster_ordering = None

        # use max replicas across all clusters
        self.share_max_replicas = share_max_replicas

        # generalize throughputs to other clusters
        self.project_throughputs = project_throughputs
        self.gput_ratios = dict()
        if self.project_throughputs:
            print(f"Project speedups active")
            self.share_max_replicas = True

        self.sched_interval = sched_interval

        # smart placement -- set to False for simulation
        # simulator does not care for changes in placement, only allocations
        self.smart_placement = False

        # default for MIP policy
        self.enable_bsz_tuning = enable_bsz_tuning

        # track prev allocs -- v2
        # self.prev_allocs['rigid'] = { job_name : cluster_idx}
        # self.prev_allocs['adaptive'] = { job_name : config_idx}
        # ngpus is number of gpus allocated for a rigid job
        # cluster_idx = None if job is not allocated any resources, else it is an index into self.cluster_ordering
        # config_idx indexes into self.idx_to_config_map, set to None if no allocation
        self.prev_allocs_v2 = {'rigid': dict(), 'adaptive': dict()}

        # large array with only ones
        self.ones_ndarray = np.ones(shape=(1, 10000), dtype=np.float32)

        # track policy metrics over time
        self.objective_values = []
        self.effective_gpus = []

    def populate_valid_configs(self, cluster_num_nodes, cluster_num_gpus, cluster_max_physical_nnodes=None):
        self.configs = dict()
        self.idx_to_config_map = dict()
        self.config_to_idx_map = dict()
        self.cluster_ordering = []
        print(f"Unique configs:")
        idx = 0
        for cluster_name in cluster_num_gpus.keys():
            self.cluster_ordering.append(cluster_name)
            nnodes, ngpus = cluster_num_nodes[cluster_name], cluster_num_gpus[cluster_name]
            nnodes = min(nnodes, cluster_max_physical_nnodes[cluster_name])
            alloc_configs = CONFIGS_4GPU if ngpus == 4 else CONFIGS_8GPU
            valid_config_idxs = alloc_configs[0] <= nnodes
            num_valid_nodes = alloc_configs[0][valid_config_idxs]
            num_valid_gpus = alloc_configs[1][valid_config_idxs]
            alloc_configs = (num_valid_nodes, num_valid_gpus)
            for nnodes, ngpus in zip(num_valid_nodes, num_valid_gpus):
                config = (cluster_name, nnodes, ngpus)
                self.idx_to_config_map[idx] = config
                self.config_to_idx_map[config] = idx
                idx += 1
            self.configs[cluster_name] = alloc_configs
        self.num_configs = {k: len(v[1]) for k, v in self.configs.items()}
        print(f"Configs: {self.idx_to_config_map}")
        # store cluster config for future use
        self.cluster_num_nodes, self.cluster_ngpus_per_node = cluster_num_nodes, cluster_num_gpus
        self.cluster_total_gpus = {
            cname: self.cluster_ngpus_per_node[cname] * self.cluster_num_nodes[cname] for cname in self.cluster_ordering}
        self.cluster_total_gpus_ndarray = np.asarray(
            [self.cluster_total_gpus[cname] for cname in self.cluster_ordering], dtype=np.uint32)
        print(f"Cluster total gpus: {self.cluster_total_gpus}")
        self.adaptive_num_replicas = np.asarray(
            [v[2] for v in self.idx_to_config_map.values()], dtype=np.uint32).reshape(-1, 1)
        self.total_num_configs = sum(self.num_configs.values())
        # matrix to project chosen config to number of GPUs
        # see v2 of solver
        self.adaptive_reduce_matrix = np.zeros(
            shape=(self.total_num_configs, len(self.cluster_ordering)), dtype=np.uint32)
        cur_idx = 0
        for i, cname in enumerate(self.cluster_ordering):
            for j in range(self.num_configs[cname]):
                config_num_gpus = self.idx_to_config_map[cur_idx][2]
                self.adaptive_reduce_matrix[cur_idx, i] = config_num_gpus
                cur_idx += 1
        # print(f"Adaptive reduce matrix: \n{self.adaptive_reduce_matrix}")

    # job_allocs: {jobname : (num_nodes, num_gpus)}
    # returns {jobname : [node_ids]} as allocations for a cluster

    def alloc_to_placement(self, cluster_name, job_allocs, prev_allocs, node_remaining_gpus):
        if self.smart_placement:
            return self.alloc_to_placement_smart(cluster_name, job_allocs, prev_allocs, node_remaining_gpus)
        else:
            return self.alloc_to_placement_fast(cluster_name, job_allocs, prev_allocs, node_remaining_gpus)

    def alloc_to_placement_fast(self, cluster_name, job_allocs, prev_allocs, node_remaining_gpus):
        max_num_nodes = len(node_remaining_gpus)
        cur_node_id = 0
        placements = {}
        # sort by num gpus needed
        job_order = sorted(list(job_allocs.items()),
                           key=lambda x: x[1][1], reverse=True)
        ngpus_per_node = np.max(node_remaining_gpus)
        # try to alloc distributed jobs on different nodes first
        for jobname, alloc in job_order:
            num_nodes, num_gpus = alloc
            node_id = cur_node_id
            num_full_nodes = num_gpus // ngpus_per_node

            # corner case
            job_placement = []
            if num_gpus == 0:
                placements[jobname] = job_placement
                continue

            # check if num_full_nodes number of nodes are available
            if num_full_nodes > 0:
                num_checked = 0
                while num_checked < max_num_nodes and num_full_nodes > 0:
                    node_gpus = node_remaining_gpus[node_id]
                    # can take full node
                    if node_gpus == ngpus_per_node:
                        node_remaining_gpus[node_id] -= ngpus_per_node
                        num_gpus -= ngpus_per_node
                        job_placement.extend([node_id] * ngpus_per_node)
                        num_full_nodes -= 1
                    node_id = (node_id + 1) % max_num_nodes
                    num_checked += 1

            # alloc any needed gpus anywhere
            while num_gpus > 0:
                node_gpus = node_remaining_gpus[node_id]
                if node_gpus != 0:
                    can_take_gpus = min(num_gpus, node_gpus)
                    num_gpus -= can_take_gpus
                    node_remaining_gpus[node_id] -= can_take_gpus
                    job_placement.extend([node_id] * can_take_gpus)
                # advance node pointer
                node_id = (node_id + 1) % max_num_nodes
            # record placement
            placements[jobname] = job_placement
            # advance cur_node_id ptr
            cur_node_id = node_id

        return placements

    def alloc_to_placement_smart(self, cluster_name, job_allocs, prev_allocs, node_remaining_gpus):
        # print(f"Cluster: {cluster_name}")
        # print(f"Allocs: {job_allocs}")
        # print(f"Prev Placements: {job_allocs}")
        max_num_nodes = len(node_remaining_gpus)
        # determined {jobname : [gpu0, gpu1, gpu2...]}
        placed_jobs = dict()
        # partition into distributed and single-node jobs
        single_node_jobs, distributed_jobs = [], []
        ngpus_per_node = max(node_remaining_gpus)
        for jobname, (nnodes, ngpus) in job_allocs.items():
            if ngpus >= ngpus_per_node:
                distributed_jobs.append(jobname)
            else:
                single_node_jobs.append(jobname)
        # preserve placements for no change in alloc
        distr_placed_jobs = dict()
        single_placed_jobs = dict()
        for jobname in job_allocs.keys():
            prev_cluster, prev_gpus = prev_allocs.get(jobname, (None, []))
            cur_nnodes, cur_ngpus = job_allocs.get(jobname, (0, 0))
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
                        for gpu_id in gpus:
                            node_remaining_gpus[gpu_id] += 1
                    assert node_remaining_gpus[reclaim_node_id] == ngpus_per_node, "eviction assert"
                    # loop again to find this freed machine
                    cur_node_id = max_num_nodes - 1
            # ensure all nodes got placed
            assert nnodes == 0, f"couldnt place -- {jobname} -> {job_allocs[jobname]}"
            distr_placed_jobs[jobname] = job_placement
        # print(f"Distributed placements: {distr_placed_jobs}")

        # alloc any single node jobs from first node ID
        def get_job_order(joblist):
            return sorted(joblist, key=lambda x: job_allocs.get(jobname, (0, 0))[1], reverse=True)
        joblist = [
            jobname for jobname in single_node_jobs if jobname not in single_placed_jobs]
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
                reclaim_ordering = sorted(
                    reclaim_cand_idxs, key=lambda x: (ngpus - node_remaining_gpus[x]))
                reclaim_node_id = reclaim_ordering[0]
                # evict some jobs from this node
                # print(f"reclaiming node --> {reclaim_node_id}")
                # find jobs mapped to this node
                reclaim_jobs = []
                for reclaim_jobname in single_placed_jobs.keys():
                    if reclaim_node_id in single_placed_jobs[reclaim_jobname]:
                        reclaim_jobs.append(reclaim_jobname)
                # sort from smallest to largest job in node
                reclaim_jobs = sorted(
                    reclaim_jobs, key=lambda x: job_allocs[x][1])
                while node_remaining_gpus[reclaim_node_id] < ngpus:
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
            assert any(
                filter), "failed to find a node to place: {jobname}; allocs = {job_allocs}, prev_allocs = {prev_allocs}, node_remaining_gpus = {node_remaining_gpus}"
            # simple packing algo -- most full valid placement
            place_idxs = idxs[filter]
            place_idxs = sorted(
                place_idxs, key=lambda x: node_remaining_gpus[x])
            place_idx = place_idxs[0]
            job_placement.extend([int(place_idx)] * ngpus)
            node_remaining_gpus[place_idx] -= ngpus
            single_placed_jobs[jobname] = job_placement
        # print(f"Single node placements: {single_placed_jobs}")
        placed_jobs = distr_placed_jobs
        placed_jobs.update(single_placed_jobs)
        return placed_jobs

    # DEPRECATED --> use _v2
    def _compute_goodputs(self, job_info, cluster_name, num_nodes, num_gpus):
        scale_unit = job_info.scale_unit[cluster_name]
        num_replicas = num_gpus // scale_unit
        speedup_fn = job_info.speedup_fn.get(cluster_name, None)
        if speedup_fn is None and not self.project_throughputs:
            return None
        if speedup_fn is not None:
            # speedup_fn exists for job in `cluster_name` cluster
            goodput_arr = np.asarray(speedup_fn.get_goodput(num_nodes.astype(
                np.float32), num_replicas.astype(np.float32)), dtype=np.float32)
            return goodput_arr

        # self.project_throughputs and speedup_fn is None:
        # check if some speedup fn is not None
        any_speedup_fn = any(
            [v is not None for v in job_info.speedup_fn.values()])
        if not any_speedup_fn:
            return None
        # take any speedup_fn
        dest_cluster = [k for k in job_info.speedup_fn.keys(
        ) if job_info.speedup_fn[k] is not None][0]
        is_dest_cluster_4gpu = self.cluster_ngpus_per_node[dest_cluster] == 4
        is_src_cluster_4gpu = self.cluster_ngpus_per_node[cluster_name] == 4

        # get scale unit for src and dest clusters
        src_scale_unit = job_info.scale_unit[cluster_name]
        dest_scale_unit = job_info.scale_unit[dest_cluster]

        # inflate/deflate num_replicas using scale_unit
        input_num_replicas = num_replicas
        input_num_nodes = num_nodes
        src_ngpus_per_node = self.cluster_ngpus_per_node[cluster_name]
        dest_ngpus_per_node = self.cluster_ngpus_per_node[dest_cluster]

        # assert that src_scale_unit is a multiple of dest_scale_unit or vice versa
        assert src_scale_unit % dest_scale_unit == 0 or dest_scale_unit % src_scale_unit == 0, \
            f"src_scale_unit {src_scale_unit} and dest_scale_unit {dest_scale_unit} are not multiples of each other"

        # translate num_replicas to dest cluster
        translated_num_replicas = input_num_replicas
        translated_num_gpus = translated_num_replicas * dest_scale_unit
        # recompute num_nodes for dest cluster
        translated_num_nodes = np.ceil(
            translated_num_gpus / dest_ngpus_per_node).astype(np.uint32)

        # remove configs that exceed cluster size
        # disabled since we could have more ngpus in src cluster than dest cluster
        '''
        max_dest_cluster_size = self.cluster_ngpus_per_node[dest_cluster] * \
            self.cluster_num_nodes[dest_cluster]
        valid_dest_configs_idxs = (translated_num_replicas <= max_dest_cluster_size) & (
            translated_num_nodes <= self.cluster_num_nodes[dest_cluster])
        '''

        # multiplier for throughput projection
        multiplier = job_info.cluster_throughput_ratios[cluster_name][dest_cluster]

        # actual speedup_fn being evaluated
        dest_speedup_fn = job_info.speedup_fn[dest_cluster]

        # match output shape to input
        output_arr = np.zeros_like(input_num_nodes)
        # translated_num_nodes = translated_num_nodes[valid_dest_configs_idxs]
        # translated_num_replicas = translated_num_replicas[valid_dest_configs_idxs]

        goodput_arr = np.asarray(dest_speedup_fn.get_goodput(translated_num_nodes.astype(
            np.float32), translated_num_replicas.astype(np.float32)), dtype=np.float32)

        # output_arr[valid_dest_configs_idxs] = goodput_arr * multiplier
        output_arr = goodput_arr * multiplier
        if sim_config.sia_log_goodputs:
            print(f"Imputing goodput for {cluster_name} from {dest_cluster}: " \
                 f"input==>({num_nodes}, {num_gpus}) -> " \
                 f"(scale_unit transform)==>({input_num_nodes}, {input_num_replicas}) -> " \
                 f"(project)==>({translated_num_nodes}, {translated_num_gpus}) -> " \
                 f"(compute goodput)==>{goodput_arr} -> " \
                 f"(scale_unit transform)==>{output_arr}")
        return output_arr
    
    # Computes goodput for adaptive job in `job_info` on `cluster_name` GPU type
    # `num_nodes` and `num_gpus` are ndarrays describing Sia configs
    # Converts nGPUs_SRC -> nScale units (number of replicas)
    # Converts nScale units -> nGPUs_DST (number of GPUs to project to in dest GPU type)
    # Computes goodput for nGPUs_DST with the right number of nnodes
    def _compute_goodputs_v2(self, job_info, cluster_name, num_nodes, num_gpus):
        scale_unit = job_info.scale_unit[cluster_name]
        num_replicas = num_gpus // scale_unit
        speedup_fn = job_info.speedup_fn.get(cluster_name, None)
        # no speedup_fn for this cluster and no project_throughputs
        if speedup_fn is None and not self.project_throughputs:
            return None
        # speedup_fn exists for this cluster --> use it
        if speedup_fn is not None:
            # speedup_fn exists for job in `cluster_name` cluster
            goodput_arr = np.asarray(speedup_fn.get_goodput(num_nodes.astype(
                np.float32), num_replicas.astype(np.float32)), dtype=np.float32)
            return goodput_arr

        # project_throughputs enabled and speedup_fn is None
        # check if some speedup fn is not None
        any_speedup_fn = any(
            [v is not None for v in job_info.speedup_fn.values()])
        # no speedup fn exists for any cluster, return raw throughput @ min-GPU on this GPU type
        if not any_speedup_fn:
            print(f"Warning: no speedup_fn exists for any cluster: returning raw throughput @ 1GPU = {job_info.cluster_throughputs[cluster_name]}")
            return job_info.cluster_throughputs[cluster_name]
        
        # take any speedup_fn
        dest_cluster = [k for k in job_info.speedup_fn.keys(
        ) if job_info.speedup_fn[k] is not None][0]
        is_dest_cluster_4gpu = self.cluster_ngpus_per_node[dest_cluster] == 4
        is_src_cluster_4gpu = self.cluster_ngpus_per_node[cluster_name] == 4

        # get scale unit for src and dest clusters (=1 for DP jobs)
        # >1 for MP, PMP jobs
        src_scale_unit = job_info.scale_unit[cluster_name]
        dest_scale_unit = job_info.scale_unit[dest_cluster]

        # inflate/deflate num_replicas using scale_unit
        input_num_replicas = num_replicas
        input_num_nodes = num_nodes
        src_ngpus_per_node = self.cluster_ngpus_per_node[cluster_name]
        dest_ngpus_per_node = self.cluster_ngpus_per_node[dest_cluster]

        # assert that src_scale_unit is a multiple of dest_scale_unit or vice versa
        assert src_scale_unit % dest_scale_unit == 0 or dest_scale_unit % src_scale_unit == 0, \
            f"src_scale_unit {src_scale_unit} and dest_scale_unit {dest_scale_unit} are not multiples of each other"

        # translate num_replicas to dest cluster
        translated_num_replicas = input_num_replicas
        translated_num_gpus = translated_num_replicas * dest_scale_unit
        # recompute num_nodes for dest cluster
        translated_num_nodes = np.ceil(
            translated_num_gpus / dest_ngpus_per_node).astype(np.uint32)

        # remove configs that exceed cluster size
        # disabled since we could have more ngpus in src cluster than dest cluster
        # for PMP jobs (does not violate any condition for DP jobs with <64 GPUs)
        '''
        max_dest_cluster_size = self.cluster_ngpus_per_node[dest_cluster] * \
            self.cluster_num_nodes[dest_cluster]
        valid_dest_configs_idxs = (translated_num_replicas <= max_dest_cluster_size) & (
            translated_num_nodes <= self.cluster_num_nodes[dest_cluster])
        '''

        # multiplier for throughput projection
        multiplier = job_info.cluster_throughput_ratios[cluster_name][dest_cluster]

        # actual speedup_fn being evaluated
        dest_speedup_fn = job_info.speedup_fn[dest_cluster]

        # match output shape to input
        output_arr = np.zeros_like(input_num_nodes)
        # translated_num_nodes = translated_num_nodes[valid_dest_configs_idxs]
        # translated_num_replicas = translated_num_replicas[valid_dest_configs_idxs]

        goodput_arr = np.asarray(dest_speedup_fn.get_goodput(translated_num_nodes.astype(
            np.float32), translated_num_replicas.astype(np.float32)), dtype=np.float32)

        # output_arr[valid_dest_configs_idxs] = goodput_arr * multiplier
        output_arr = goodput_arr * multiplier
        if sim_config.sia_log_goodputs:
            print(f"Imputing goodput for {cluster_name} from {dest_cluster}: " \
                 f"input==>({num_nodes}, {num_gpus}) -> " \
                 f"(scale_unit transform)==>({input_num_nodes}, {input_num_replicas}) -> " \
                 f"(project)==>({translated_num_nodes}, {translated_num_gpus}) -> " \
                 f"(compute goodput)==>{goodput_arr} -> " \
                 f"(scale_unit transform)==>{output_arr}")
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
                    gput_ratios.setdefault(dst_cluster, dict())[
                        src_cluster] = ratio
        self.gput_ratios = gput_ratios

    def get_current_gput_ratios(self):
        return self.gput_ratios if self.gput_ratios else dict()

    # computes goodputs and re-alloc factor for the rigid job in each GPU type
    def _compute_rigid_job_goodput_single(self, rigid_job, rigid_jobname):
        # rigid job goodput matrix: 1 x gpu_types
        goodputs = []
        ngpus = max(rigid_job.max_replicas.values())

        # pick up throughputs from job_info.cluster_throughput_ratios
        # goodput = throughput since bsz is fixed
        goodputs = []
        for cname in self.cluster_ordering:
            goodputs.append(rigid_job.cluster_throughput_ratios[cname])

        # normalize and scale goodputs so that slowest cluster has goodput = ngpus
        min_goodput = min(goodputs)
        scale_factor = ngpus / min_goodput
        goodputs = [g * scale_factor for g in goodputs]

        # compute rescale factor
        # re-alloc/migrate factor
        job_restart_penalty = get_restart_penalty(rigid_jobname)
        job_lost_gpu_seconds = (
            rigid_job.num_restarts * job_restart_penalty)
        realloc_factor = max(
            (rigid_job.age - job_lost_gpu_seconds), 0) / (rigid_job.age + job_restart_penalty)
        realloc_factor = 1 if rigid_job.attained_service < job_restart_penalty else realloc_factor
        print(
            f"Rigid job: {rigid_jobname}, ngpus: {ngpus}, goodputs: {goodputs}, realloc_factor: {realloc_factor}")
        return goodputs, realloc_factor

    # computes goodput and damping factors for rigid jobs:
    # * `rigid_goodputs` (normalized to slowest cluster, with slowest goodput = num replicas)
    #                   -- goodput for each rigid job in each GPU type (num_jobs x num_gpu_types)
    #                   -- ordering defined by self.cluster_ordering
    # * `rigid_realloc_factors` -- realloc factor for each rigid job
    # * `rigid_migrate_factors` -- migrate factor for each rigid job
    def _get_rigid_goodputs(self, rigid_joblist, rigid_jobnames):
        # normalized goodput matrix for rigid jobs: num_jobs x num_gpu_types
        rigid_goodputs = np.zeros(
            (len(rigid_joblist), len(self.cluster_ordering)))
        # number of replicas for each rigid job
        rigid_num_replicas = np.asarray(
            [max(job.max_replicas.values()) for job in rigid_joblist], dtype=np.uint32).reshape(-1, 1)
        # re-alloc factors for each rigid job
        rigid_realloc_factors = np.zeros(len(rigid_joblist))

        for i, rigid_job in enumerate(rigid_joblist):
            rigid_goodputs[i, :], rigid_realloc_factors[i] = self._compute_rigid_job_goodput_single(
                rigid_job, rigid_jobnames[i])
        return rigid_goodputs, rigid_num_replicas, rigid_realloc_factors

    def optimize_mip(self, jobs, nodes, prev_allocations):
        # print(f"Prev allocations -- {prev_allocations}")
        np.set_printoptions(suppress=True)
        adaptive_joblist, adaptive_jobnames = [], []
        rigid_joblist, rigid_jobnames = [], []
        # filter out rigid jobs, keep only scale-gpu + adaptive jobs
        for i, (jobname, job) in enumerate(jobs.items()):
            if job.category == "rigid":
                rigid_joblist.append(job)
                rigid_jobnames.append(jobname)
            else:
                adaptive_joblist.append(job)
                adaptive_jobnames.append(jobname)
        adaptive_num_jobs = len(adaptive_joblist)
        rigid_num_jobs = len(rigid_joblist)
        num_gpus = {}

        # filter jobs to only contain active jobs first
        for k, k_nodes in nodes.items():
            num_gpus[k] = 0
            for node_idx, node in k_nodes.items():
                num_gpus[k] += node.resources["nvidia.com/gpu"]
        total_num_configs = sum(self.num_configs.values())

        # single-cluster speedup-matrix
        def get_zero_matrix_adaptive(cluster_name):
            return np.zeros((adaptive_num_jobs, self.num_configs[cluster_name]), dtype=np.float32) + ZERO_ALLOC_GAIN
        cluster_goodput_matrices = {
            k: get_zero_matrix_adaptive(k) for k in self.cluster_ordering}
        realloc_factors, migrate_factors = [], []

        # job weights: equal weight for all jobs; modify if needed
        job_weights = np.ones(
            (1, adaptive_num_jobs + rigid_num_jobs), dtype=np.float32)

        # compute raw speedup matrix (irrespective of slow/fast cluster)
        for i, job in enumerate(adaptive_joblist):
            jobname = adaptive_jobnames[i]
            if sim_config.sia_log_goodputs:
                print(f"-------- Processing Adaptive Job: {jobname} --------")
            # compute _fair_ goodput
            nnz_speedups = dict()
            for cluster in self.cluster_ordering:
                speedup_fn = job.speedup_fn[cluster]
                # additional check for PMP models
                cluster_scale_unit = job.scale_unit[cluster]
                if self.share_max_replicas:
                    max_replicas = max(job.max_replicas.values())
                    min_replicas = min(job.min_replicas.values())
                else:
                    max_replicas = job.max_replicas[cluster]
                    min_replicas = job.min_replicas[cluster]
                # alloc at least `scale_unit` number of replicas
                min_replicas = max(1, min_replicas)
                max_replicas = max(1, max_replicas)
                # corner-case for PMP jobs (simulating GPT jobs using approximate profiles from BERT jobs)
                if "gpt-pmp" in jobname and (max_replicas == 6) and cluster == "dgx-ext":
                    max_replicas = 8
                if sim_config.sia_log_goodputs:
                    print(f"Cluster: {cluster}, min_replicas: {min_replicas}, max_replicas: {max_replicas}")

                # default code-path: simulates Sia not knowing 1-GPU throughputs before running
                # for the firs time
                if not sim_config.sia_read_throughputs_before_first_run:
                    if speedup_fn is None:
                        if not self.project_throughputs:
                            nnz_speedups[cluster] = cluster_scale_unit
                            continue
                        # check if any throughput model exists to extrapolate from
                        any_speedup_fn = any(
                            [fn is not None for fn in job.speedup_fn.values()])
                        if not any_speedup_fn:
                            # use raw throughputs
                            nnz_speedups[cluster] = cluster_scale_unit
                            continue

                # cluster-specific configs
                alloc_num_nodes, alloc_num_gpus = self.configs[cluster]
                valid_configs = (alloc_num_gpus <= (max_replicas * cluster_scale_unit)) & (
                    alloc_num_gpus >= (min_replicas * cluster_scale_unit))
                # additional check for PMP models
                if cluster_scale_unit > 1:
                    valid_configs = valid_configs & (
                        alloc_num_gpus % cluster_scale_unit == 0)

                valid_nnodes, valid_ngpus = alloc_num_nodes[valid_configs], alloc_num_gpus[valid_configs]
                if sim_config.sia_log_goodputs:
                    print(f"Valid_configs:: {list(zip(valid_nnodes, valid_ngpus))}")
                goodput_matrix = cluster_goodput_matrices[cluster]
                valid_configs_goodput = self._compute_goodputs_v2(
                    job, cluster, valid_nnodes, valid_ngpus)
                if sim_config.sia_log_goodputs:
                    print(f"Valid configs goodputs:: {valid_configs_goodput}")
                goodput_matrix[i, valid_configs] = valid_configs_goodput
                if valid_nnodes.size == 0:
                    nnz_speedups[cluster] = cluster_scale_unit
            # print(f"nnz speedups: {nnz_speedups}")
            # fill in min-GPU config for each cluster with lowest value for min-GPU config in any cluster
            cluster_min_goodputs = []
            for cluster in self.cluster_ordering:
                if cluster not in nnz_speedups:
                    nnz_valid_idxs = cluster_goodput_matrices[cluster][i,:] > ZERO_ALLOC_GAIN
                    cluster_min_goodputs.append(
                        np.min(cluster_goodput_matrices[cluster][i, :][nnz_valid_idxs]))
            cluster_min_goodputs = np.asarray(cluster_min_goodputs)
            if cluster_min_goodputs.size == 0:
                min_goodput = 1
            else:
                # take lowest non-zero goodput value
                min_goodput = np.min(cluster_min_goodputs)

            # normalize all goodput values to min_goodput
            scale_factor = (1.0 / min_goodput)
            # scale unit in cluster with min goodput
            min_goodput_scale_unit = 1
            # print(f"Scaling {jobname} by {scale_factor}")
            for cluster in self.cluster_ordering:
                # check if this cluster has the min goodput
                if cluster not in nnz_speedups:
                    if np.any(cluster_goodput_matrices[cluster][i, :] == min_goodput):
                        min_goodput_scale_unit = job.scale_unit[cluster]
            # ensure that min value in row == valid scale unit
            scale_factor *= min_goodput_scale_unit
            for cluster in self.cluster_ordering:
                # check if this cluster has the min goodput
                if cluster not in nnz_speedups:
                    if np.any(cluster_goodput_matrices[cluster][i, :] == min_goodput):
                        min_goodput_scale_unit = job.scale_unit[cluster]
                # normalize every config to min goodput
                cluster_goodput_matrices[cluster][i, :] *= scale_factor
                # get cluster scale unit
                cluster_scale_unit = job.scale_unit[cluster]
                # if goodput vector is empty, set first config to be [cluster_scale_unit]
                if cluster in nnz_speedups:
                    # get config idx that corresponds to `scale_unit` replicas
                    scale_unit_selector = (
                        self.configs[cluster][1] == cluster_scale_unit)
                    # set goodput to cluster_scale_unit
                    cluster_goodput_matrices[cluster][i,
                                                      :][scale_unit_selector] = cluster_scale_unit
                # assert max(cluster_goodput_matrices[cluster][i, :]) < 500, "bad speedup values"
                # fix instability from very small values by setting to ZERO_ALLOC_GAIN
                very_small_values = cluster_goodput_matrices[cluster] <= ZERO_ALLOC_GAIN
                cluster_goodput_matrices[cluster][very_small_values] = ZERO_ALLOC_GAIN

            # re-alloc/migrate factor
            job_restart_penalty = get_restart_penalty(jobname)
            if sim_config.sia_use_pollux_realloc_factor:
                # seconds lost in checkpoint+restore
                job_lost_seconds = (job.num_restarts * job_restart_penalty)
                realloc_factor = max((job.age - job_lost_seconds), 0) / (job.age + job_restart_penalty)
                print(f"Realloc factor: {jobname} --> {realloc_factor}, age: {job.age}, restart penalty: {job_restart_penalty}, lost seconds: {job_lost_seconds}, num restarts: {job.num_restarts}")
            else:
                # GPU seconds used, wasted
                used_gpu_sec = job.used_gpu_seconds
                wasted_gpu_sec = job.wasted_gpu_seconds
                gpu_sec_sum = used_gpu_sec + wasted_gpu_sec
                # ratio of used to allocated GPU seconds
                # realloc_factor = max(job.age + used_gpu_sec, 0) / (job.age + gpu_sec_sum + job_restart_penalty)
                realloc_factor = max(1 + used_gpu_sec, 0) / (1 + gpu_sec_sum + job_restart_penalty)
                print(f"Realloc factor: {jobname} --> {realloc_factor}, age: {job.age}, used: {used_gpu_sec}, wasted: {wasted_gpu_sec}, restart penalty: {job_restart_penalty}")
            
            # Don't penalize jobs that have not scaled up even once
            realloc_factor = 1 if job.age < job_restart_penalty else realloc_factor
            # realloc_factor = max(realloc_factor, 0.8)
            if realloc_factor < 0.2:
                print(f"WARNING:: low realloc factor: {jobname} --> {realloc_factor}")
                print(f">>>>>>>>>>>>>>>>>>>>>>Check if the job is restarting too often<<<<<<<<<<<<<<<<<<<<<<<<<")

            # cost of migrating job between clusters = cost of restarting job in current cluster
            migrate_factor = realloc_factor
            realloc_factors.append(realloc_factor)
            migrate_factors.append(migrate_factor)

        # one speedup matrix for all adaptive jobs across clusters
        adaptive_speedup_mat = np.hstack(
            tuple([cluster_goodput_matrices[cluster] for cluster in self.cluster_ordering]))

        if sim_config.sia_log_goodputs:
            np.set_printoptions(precision=2, linewidth=160)
            print(f"Adaptive speedup matrix: \n{repr(adaptive_speedup_mat)}")
            print(f"Re-allocation factors: {realloc_factors}")

        # get rigid job info
        rigid_speedup_mat, rigid_num_replicas, rigid_realloc_factors = self._get_rigid_goodputs(
            rigid_joblist, rigid_jobnames)

        # hard constraint:: do not re-allocate a job if it is currently doing a checkpoint-restore operation
        for i, jobname in enumerate(rigid_jobnames):
            # not currently doing a checkpoint-restore operation
            if not rigid_joblist[i].scaling_underway:
                continue
            # only one non-zero value in row i --> GPU type on which it is currently running
            job_gpu_type = self.prev_allocs_v2['rigid'][jobname]
            preserve_val = rigid_speedup_mat[i, job_gpu_type]
            rigid_speedup_mat[i, :] = 0
            rigid_speedup_mat[i, job_gpu_type] = preserve_val
            rigid_realloc_factors[i] = 1.0
        if sim_config.sia_log_goodputs:
            print(f"Rigid goodputs: {list(zip(rigid_jobnames, rigid_speedup_mat, rigid_num_replicas, rigid_realloc_factors))}")

        # hard constraint:: do not re-allocate a job if it is currently doing a checkpoint-restore operation
        # treat it as a non-preemptible job
        non_preemptible_jobnames = dict()
        non_preemptible_jobnames['rigid'] = [rigid_jobnames[i] for i in range(
            rigid_num_jobs) if rigid_joblist[i].scaling_underway]
        non_preemptible_jobnames['adaptive'] = [adaptive_jobnames[i] for i in range(
            adaptive_num_jobs) if adaptive_joblist[i].scaling_underway]
        
        print(f"Preserving allocations for jobs blocked on checkpoint-restore: {non_preemptible_jobnames}")

        # create input to optimizer: (jobnames, speedup matrix, num replicas, realloc factors)
        optim_params = {'rigid': (rigid_jobnames, rigid_speedup_mat, rigid_num_replicas, rigid_realloc_factors),
                        'adaptive': (adaptive_jobnames, adaptive_speedup_mat, realloc_factors)}
        # print optim problem inputs
        if sim_config.sia_log_problem:
            print(f"Optim input:")
            print(f"Rigid jobs: {optim_params['rigid']}")
            print(
                f"Adaptive jobs: {optim_params['adaptive'][0]}\n{optim_params['adaptive'][1]}\n realloc factors: {optim_params['adaptive'][2]}")

        job_allocs, cluster_allocs = self.__solve_mip_rescaled(optim_params, num_gpus, 
                                                               job_weights, non_preemptible_jobnames)

        # cluster-specific job placements
        cluster_job_placements = dict()
        for cluster in self.cluster_ordering:
            node_remaining_gpus = np.asarray(
                [node.resources["nvidia.com/gpu"] for idx, node in nodes[cluster].items()], dtype=np.uint32)
            if cluster in cluster_allocs:
                prev_allocs = dict()
                for job in cluster_allocs[cluster].keys():
                    prev_allocs[job] = prev_allocations.get(job, (None, []))
                cluster_job_placements[cluster] = self.alloc_to_placement(
                    cluster, cluster_allocs[cluster], prev_allocs, node_remaining_gpus)
            else:
                cluster_job_placements[cluster] = dict()

        # merge allocs
        job_placements = {}
        for k, v in job_allocs.items():
            if v is None:
                job_placements[k] = (None, ())
            else:
                cluster_name, alloc = v
                job_placements[k] = (
                    cluster_name, cluster_job_placements[cluster_name][k])
        # log placements to stdout
        # print(f"Placements: {job_placements}")

        # compute diffs between allocations
        bad_evict = 0
        bad_evictions = []
        good_evict = 0
        for job in job_placements.keys():
            old_gpus = prev_allocations.get(job, (None, []))
            new_gpus = job_placements.get(job, (None, []))
            if len(old_gpus[1]) == len(new_gpus[1]) and old_gpus[0] == new_gpus[0]:
                if old_gpus != new_gpus:
                    bad_evict += 1
                    bad_evictions.append((old_gpus, new_gpus))
            else:
                good_evict += 1
        # print(f"Evictions: good = {good_evict}, bad = {bad_evict} ==> {bad_evictions}")
        return job_placements

    # speedups scaled for reallocation
    # cluster_num_gpus = (num_slow_gpus, num_fast_gpus)
    def __solve_mip_rescaled(self, optim_params, free_gpus, job_weights, non_preemptible_jobnames):
        # started setting up optimization problem
        st_time = time.time()
        # extract params
        rigid_params, adaptive_params = optim_params['rigid'], optim_params['adaptive']
        rigid_jobnames, rigid_speedup_mat, rigid_num_replicas, rigid_realloc_factors = rigid_params
        adaptive_jobnames, adaptive_speedup_mat, adaptive_realloc_factors = adaptive_params
        rigid_num_jobs, rigid_gpu_types = rigid_speedup_mat.shape
        adaptive_num_jobs, adaptive_num_configs = adaptive_speedup_mat.shape
        adaptive_weights, rigid_weights = job_weights[:,
                                                      :adaptive_num_jobs], job_weights[:, adaptive_num_jobs:]
        # degenerate case
        if adaptive_num_jobs + rigid_num_jobs == 0:
            return dict(), dict()
        
        # make copy of adaptive_speedup_matrix and rigid_speedup_matrix
        adaptive_speedup_mat_copy = np.copy(adaptive_speedup_mat)
        rigid_speedup_mat_copy = np.copy(rigid_speedup_mat)

        # ones vectors : N x 1 size, use arr.T if you want 1 x N
        ones_rigid_num_jobs = np.ones(
            shape=(rigid_num_jobs, 1), dtype=np.uint32)
        ones_adaptive_num_jobs = np.ones(
            shape=(adaptive_num_jobs, 1), dtype=np.uint32)
        ones_num_configs = np.ones(
            shape=(adaptive_num_configs, 1), dtype=np.uint32)
        ones_num_gpu_types = np.ones(
            shape=(rigid_gpu_types, 1), dtype=np.uint32)

        # rescale speedups for rigid jobs
        # speedup[job, gpu_type] *= realloc_factor[job] if (prev_alloc[job] != gpu_type)
        for i in range(rigid_num_jobs):
            prev_gpu_type = self.prev_allocs_v2['rigid'].get(
                rigid_jobnames[i], None)
            # has a previous allocation
            if prev_gpu_type is not None:
                # scale down speedups for all other gpu types, except the previous allocation
                preserve_val = rigid_speedup_mat[i, prev_gpu_type]
                rigid_speedup_mat[i, :] *= rigid_realloc_factors[i]
                rigid_speedup_mat[i, prev_gpu_type] = preserve_val
            # job must present some valid GPU type to schedule on
            has_valid_schedule = np.all(rigid_speedup_mat[i, :] >= 0) and np.sum(
                rigid_speedup_mat[i, :]) > 0
            assert has_valid_schedule or (
                rigid_jobnames[i] in non_preemptible_jobnames['rigid']), f"missing speedup values for job: {rigid_jobnames[i]}"

        # rescale speedups for adaptive jobs
        # speedup[job, config_idx] *= realloc_factor[job] if (prev_alloc[job] != config_idx)
        for i in range(adaptive_num_jobs):
            prev_config_idx = self.prev_allocs_v2['adaptive'].get(
                adaptive_jobnames[i], None)
            # has a previous allocation
            if prev_config_idx is not None:
                # scale down speedups for all other gpu types, except the previous allocation
                preserve_val = adaptive_speedup_mat[i, prev_config_idx]
                adaptive_speedup_mat[i, :] *= adaptive_realloc_factors[i]
                adaptive_speedup_mat[i, prev_config_idx] = preserve_val
            # job must present some valid config to schedule on
            has_valid_schedule = np.all(adaptive_speedup_mat[i, :] >= 0) and np.sum(
                adaptive_speedup_mat[i, :]) > 0
            assert has_valid_schedule or (
                adaptive_jobnames[i] in non_preemptible_jobnames['adaptive']), f"missing speedup values for job: {adaptive_jobnames[i]}"

        # power up speedup matrices
        def __power_up_speedup_matrix(speedup_matrix, p_fairness):
            # power up speedup matrix to make it more fair
            pup_mat = np.power(speedup_matrix, self.p_fairness)
            pup_mat = np.round(pup_mat, decimals=2)
            # cap max value : sim_config.sia_goodput_clip_val
            clip_threshold_val = sim_config.sia_goodput_clip_val
            pup_mat[pup_mat > clip_threshold_val] = clip_threshold_val
            '''
            # at least one element in the array
            if speedup_matrix.shape[0] > 0:
                print(f"Maximum speedup value in matrix: {np.max(pup_mat)}")
                assert np.max(
                    pup_mat) < 250, f"max value in speedup mat exceeds 250 (ignore if alright). Matrix as follows\n{pup_mat}"
            '''
            return pup_mat

        if adaptive_num_jobs > 0:
            adaptive_A = __power_up_speedup_matrix(
                adaptive_speedup_mat, self.p_fairness)
        else:
            adaptive_A = 0
        if rigid_num_jobs > 0:
            rigid_A = __power_up_speedup_matrix(
                rigid_speedup_mat, self.p_fairness)
        else:
            rigid_A = 0

        # regularization parameters
        opt_lambda_alloc_change = self.lambda_a if self.lambda_a else -0.02
        opt_lambda_no_alloc = self.lambda_n if self.lambda_n else -1

        # construct variable to optimize over
        if adaptive_num_jobs > 0:
            adaptive_X = cp.Variable(shape=adaptive_A.shape, integer=True)
        else:
            adaptive_X = 0
        if rigid_num_jobs > 0:
            rigid_X = cp.Variable(shape=rigid_A.shape, integer=True)
        else:
            rigid_X = 0

        ## OBJECTIVE ##
        # construct base sia objective
        constraints = []
        # sum goodput over chosen configs
        obj_expr = cp.sum(adaptive_weights @ cp.multiply(adaptive_X,
                          adaptive_A)) if adaptive_num_jobs > 0 else 0
        obj_expr += cp.sum(rigid_weights @ cp.multiply(rigid_X,
                           rigid_A)) if rigid_num_jobs > 0 else 0

        # penalize num allocations < num jobs (or 1 for each unallocated job)
        # note: MUST NOT be made a constraint since it is possible to have num_jobs > num_gpus
        #       in which case the constraint will be violated, despite having a realizable schedule
        num_allocs = cp.sum(adaptive_X) if adaptive_num_jobs > 0 else 0
        num_allocs += cp.sum(rigid_X) if rigid_num_jobs > 0 else 0

        num_jobs = adaptive_num_jobs + rigid_num_jobs
        obj_expr += opt_lambda_no_alloc * (num_jobs - num_allocs)

        # finish constructing objective
        if self.p_fairness < 0:
            obj = cp.Minimize(obj_expr)
        else:
            obj = cp.Maximize(obj_expr)

        ## CONSTRAINTS ##
        # constrain range of X matrices: X = {0, 1} for each job
        if adaptive_num_jobs > 0:
            constraints.append(adaptive_X >= 0)
            constraints.append(adaptive_X <= 1)
        if rigid_num_jobs > 0:
            constraints.append(rigid_X >= 0)
            constraints.append(rigid_X <= 1)

        # per-cluster GPU count constraint
        # num_gpus_used_vec: size = (num_gpu_types, 1)
        num_gpus_used_vec = 0
        if rigid_num_jobs > 0:
            # rigid_num_replicas: size = (num_rigid_jobs, 1)
            # rigid_reduce_matrix: size = (num_rigid_jobs, num_gpu_types)
            # construct a matrix with all columns the same as rigid_num_replicas
            rigid_reduce_matrix = np.tile(
                rigid_num_replicas, (1, rigid_gpu_types))

            # rigid_X : size = (num_rigid_jobs, num_gpu_types)
            # rigid_used_vec: size = (num_rigid_jobs, num_gpu_types)
            rigid_used_vec = cp.multiply(rigid_X, rigid_reduce_matrix)
            # accumulate rigid_used_vec into num_gpus_used_vec
            num_gpus_used_vec += cp.sum(rigid_used_vec, axis=0)
        if adaptive_num_jobs > 0:
            # number of gpus used for each job
            # adaptive_gpus_used_vec: size = (num_adaptive_jobs, num_gpu_types)
            adaptive_gpus_used_vec = adaptive_X @ self.adaptive_reduce_matrix
            # accumulate adaptive_gpus_used_vec into num_gpus_used_vec
            # by summing over each adaptive job
            num_gpus_used_vec += cp.sum(adaptive_gpus_used_vec, axis=0)

        # add constraint for total num gpus used per GPU type
        constraints.append(num_gpus_used_vec <=
                           self.cluster_total_gpus_ndarray)
        constraints.append(num_gpus_used_vec >= 0)

        # constraint only one config per job for adaptive and rigid jobs
        # also, force preserve allocation for jobs in non_preemptible_jobs
        if rigid_num_jobs > 0:
            constraints.append((rigid_X @ ones_num_gpu_types)
                               <= ones_rigid_num_jobs)
            for jobname in non_preemptible_jobnames['rigid']:
                job_idx = rigid_jobnames.index(jobname)
                job_gpu_type = self.prev_allocs_v2['rigid'].get(jobname, None)
                assert job_gpu_type is not None, f"job not found in prev_allocs: {jobname}, prev_allocs: {self.prev_allocs_v2}"
                constraints.append(rigid_X[job_idx, job_gpu_type] == 1)
        if adaptive_num_jobs > 0:
            constraints.append((adaptive_X @ ones_num_configs)
                               <= ones_adaptive_num_jobs)
            for jobname in non_preemptible_jobnames['adaptive']:
                job_idx = adaptive_jobnames.index(jobname)
                job_config_idx = self.prev_allocs_v2['adaptive'].get(
                    jobname, None)
                assert job_config_idx is not None, f"job not found in prev_allocs: {jobname}, prev_allocs: {self.prev_allocs_v2}"
                constraints.append(adaptive_X[job_idx, job_config_idx] == 1)
        # solve problem
        problem = cp.Problem(obj, constraints=constraints)
        st_time2 = time.time()
        # problem.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF'}, 
        #               verbose=sim_config.sia_solver_verbose)
        problem.solve(solver=cp.CBC, verbose=sim_config.sia_solver_verbose, 
                      numberThreads=sim_config.sia_solver_num_threads, 
                      allowablePercentageGap=sim_config.sia_solver_thresh*100)
        ed_time = time.time()
        # print(f"OPTIM: Problem: {problem}")
        solver_setup_time = (st_time2 - st_time)*1000
        solver_solve_time = (ed_time - st_time2)*1000
        solver_total_time = (ed_time - st_time)*1000
        if problem.status == 'optimal':
            print(f"Solver took: {solver_total_time:.2f}ms = {solver_setup_time:.2f}ms (setup) + {solver_solve_time:.2f}ms (solve)")
        if problem.status != 'optimal':
            print(f"Solver time: {ed_time - st_time}s")
            print("Status: ", problem.status)
            print(f"Problem: {problem}")
            print("The optimal value is", problem.value)
            print("A solution x is")
            print(rigid_X.value)
            print(adaptive_X.value)
        if rigid_num_jobs > 0:
            # print(f"Solution == rigid_X: {rigid_X.value}")
            pass
        if adaptive_num_jobs > 0:
            # print(f"adaptive_X: {adaptive_X.value}")
            pass
        # rigid_allocs, adaptive_allocs: jobname -> (cluster, num_nodes, num_replicas)
        rigid_allocs, adaptive_allocs = dict(), dict()
        # cluster_allocs: cluster -> jobname -> (num_nodes, num_replicas)
        cluster_allocs = dict()
        # track metrics for this round
        round_effective_ngpus = dict()
        self.objective_values.append(problem.value)
        # turn solution into allocs for rigid jobs
        if rigid_num_jobs > 0:
            # round solution
            rigid_output_soln = np.round(
                rigid_X.value, decimals=0).astype(np.uint32)
            # convert 1-hot alloc vector to GPU type IDX
            chosen_gpu_types = np.argmax(rigid_output_soln, axis=1)
            # process allocs for rigid jobs
            for i, jobname in enumerate(rigid_jobnames):
                alloc_id = None
                # if no alloc, set to None
                if np.sum(rigid_output_soln[i, :]) == 0:
                    rigid_allocs[jobname] = None
                    round_effective_ngpus[jobname] = 0
                else:
                    # get cluster name
                    gpu_type = alloc_id = chosen_gpu_types[i].item()
                    cname = self.cluster_ordering[gpu_type]
                    if cname not in cluster_allocs:
                        cluster_allocs[cname] = dict()
                    num_replicas = rigid_num_replicas[i].astype(
                        np.uint32).item()
                    num_nodes = np.ceil(
                        num_replicas * 1.0 / self.cluster_ngpus_per_node[cname]).astype(np.uint32).item()
                    rigid_allocs[jobname] = (cname, (num_nodes, num_replicas))
                    cluster_allocs[cname][jobname] = (num_nodes, num_replicas)
                    # record effective ngpus allocated for this job
                    round_effective_ngpus[jobname] = rigid_speedup_mat_copy[i, alloc_id]
                # update prev_allocs for this job for next round
                self.prev_allocs_v2['rigid'][jobname] = alloc_id
        if adaptive_num_jobs > 0:
            # round solution
            adaptive_output_soln = np.round(
                adaptive_X.value, decimals=0).astype(np.uint32)
            # convert 1-hot alloc vector to config IDX
            chosen_configs = np.argmax(adaptive_output_soln, axis=1)
            # process allocs for adaptive jobs
            for i, jobname in enumerate(adaptive_jobnames):
                alloc_id = None
                # if no alloc, set to None
                if np.sum(adaptive_output_soln[i, :]) == 0:
                    adaptive_allocs[jobname] = None
                    round_effective_ngpus[jobname] = 0
                else:
                    # get config using chosen config idx
                    config_idx = alloc_id = chosen_configs[i].item()
                    cname, num_nodes, num_replicas = self.idx_to_config_map[config_idx]
                    adaptive_allocs[jobname] = (
                        cname, (num_nodes, num_replicas))
                    if cname not in cluster_allocs:
                        cluster_allocs[cname] = dict()
                    cluster_allocs[cname][jobname] = (num_nodes, num_replicas)
                    # record effective ngpus allocated for this job
                    round_effective_ngpus[jobname] = adaptive_speedup_mat_copy[i, alloc_id]
                # update prev_allocs for this job for next round
                self.prev_allocs_v2['adaptive'][jobname] = alloc_id

        # append round effective ngpus to list
        self.effective_gpus.append(round_effective_ngpus)

        # filter self.prev_allocs to remove jobs that are no longer in the queue
        active_jobnames = set(rigid_jobnames + adaptive_jobnames)
        self.prev_allocs_v2['rigid'] = {k: v for k, v in self.prev_allocs_v2['rigid'].items(
        ) if k in active_jobnames}
        self.prev_allocs_v2['adaptive'] = {k: v for k, v in self.prev_allocs_v2['adaptive'].items(
        ) if k in active_jobnames}

        # merge rigid and adaptive allocs into one dict
        job_allocs = {**rigid_allocs, **adaptive_allocs}

        # print metrics for round
        print(f"Optimization metrics: optimal objective value = {problem.value:.4f}, effective ngpus = {sum(round_effective_ngpus.values())}")

        return job_allocs, cluster_allocs

    def optimize(self, jobs, nodes, base_allocations):
        assert self.p_fairness != 0, f"Invalid p value : {self.p_fairness}"
        return self.optimize_mip(jobs, nodes, base_allocations)
