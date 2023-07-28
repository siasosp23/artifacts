import collections
import copy
import json
import csv
import time

import numpy as np
from collections import Counter

from applications import APPLICATIONS
from pollux_unaware import PolluxPolicyUnaware
from sia_unaware import SiaUnaware
from utils import JobInfoUnaware, NodeInfo
from job import Job
from pollux_unaware_fix import PolluxPolicyUnawareFix
## WARNING:: Shockwave is currently disabled as it requires Gurobi license
#from shockwave import ShockWavePolicy
ShockWavePolicy=dict


class ClusterUnaware(object):
    # num_nodes, ngpus_per_node are dicts with key=cluster_name
    def __init__(self, workload, policy, num_nodes, ngpus_per_node, max_num_replicas=None, max_physical_nodes=None):
        self.workload = workload
        self.policy = policy
        self.num_nodes = num_nodes
        self.num_gpus = ngpus_per_node
        self.current_time = 0
        self.clusters = list(self.num_nodes.keys())
        self.max_num_replicas = max_num_replicas
        self.max_physical_nodes = max_physical_nodes

        # create inverse map like {app-name: {cluster_name: cluster_app}}
        self.cluster_applications = dict()
        for cluster_name, cluster_apps in APPLICATIONS.items():
            for app_name, app in cluster_apps.items():
                if app_name not in self.cluster_applications:
                    self.cluster_applications[app_name] = dict()
                self.cluster_applications[app_name][cluster_name] = app

        # create a static map of v-node ids to real node ids
        vnode_id = 0
        self.vnode_cluster_node_id_map = dict()
        for cluster in self.clusters:
            num_vnodes_per_node = int(
                self.num_gpus[cluster] / min(self.num_gpus.values()))
            for cluster_node_id in range(self.num_nodes[cluster]):
                for _ in range(num_vnodes_per_node):
                    self.vnode_cluster_node_id_map[vnode_id] = (
                        cluster, cluster_node_id)
                    vnode_id += 1
        if isinstance(policy, SiaUnaware) or isinstance(policy, PolluxPolicyUnaware):
            cache_speedups = isinstance(policy, PolluxPolicyUnaware)
            self.jobs = [Job(row.name, self.cluster_applications[row.application],
                             row.time, cache_speedups=cache_speedups, h_unaware=True) for row in workload.itertuples()]
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
        elif isinstance(policy, PolluxPolicyUnawareFix) or isinstance(policy, ShockWavePolicy):
            min_cluster_size = min(
                [self.num_nodes[k] * self.num_gpus[k] for k in self.num_nodes.keys()])
            cache_speedups = isinstance(policy, PolluxPolicyUnawareFix)
            self.target_num_replicas = [
                min(row.num_replicas, min_cluster_size) for row in workload.itertuples()]
            self.jobs = [Job(row.name, self.cluster_applications[row.application],
                             row.time, cache_speedups=cache_speedups, h_unaware=True, target_num_replicas=min(row.num_replicas, min_cluster_size),
                             target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
            for job in self.jobs:
                job.fix_minibatch_size()
        else:
            assert False, f"unsupported policy {policy.__class__.__name__}"

        self.allocations = {}
        self.vnode_allocations = {}
        self.wasted_gpus = []
        self.logs = []
        self.utility = []

        # set cluster wide max replicas = max(num gpus in any cluster)
        # self.clusterwide_max_replicas = int(min(max(
        #    [self.num_nodes[cluster] * self.num_gpus[cluster] for cluster in self.num_nodes.keys()]), 64))
        self.clusterwide_max_replicas = max_num_replicas
        print(f"Max scale for jobs: {self.clusterwide_max_replicas}")

        # shockwave add-ons
        self.active_jobs = []

    def step(self, seconds=60):
        start_t = time.time()
        self.step_jobs(seconds)
        end_t = time.time()
        print(f"Job Step time: {(end_t - start_t)*1000}ms")
        self.optimize_policy()
        end_t2 = time.time()
        print(f"Scheduling compute time: {(end_t2 - end_t)*1000}ms")
        self.step_log()

    def step_jobs(self, seconds=60):
        check_interference = False
        # TODO: fix interference check code
        if check_interference:
            interfere_nodes = set(idx for idx in range(self.num_nodes)
                                  if sum(len(set(val)) > 1 and idx in val
                                         for key, val in self.allocations.items()) > 1)
        for i, job in enumerate(self.jobs):
            if check_interference:
                alloc_set = set(self.allocations.get(job.name, [])[1])
                interference = 0.0 if len(alloc_set) > 1 and any(
                    idx in interfere_nodes for idx in alloc_set) else 0.0
            else:
                interference = 0.0
            job.step(seconds, interference=interference)

            #  check batch size and num_gpus
            if isinstance(self.policy, PolluxPolicyUnawareFix) or isinstance(self.policy, ShockWavePolicy):
                # bsz
                error = 1 * (job.accum_steps + 1) * int(sum(job.placement))
                actual_bsz = job.atomic_bsz * \
                    (job.accum_steps + 1) * int(sum(job.placement))
                assert actual_bsz == 0 or abs(
                    actual_bsz - job.target_batch_size) <= error, f"actual: {actual_bsz}, target: {job.target_batch_size}"

                # num gpus
                assert job.target_num_replicas == self.target_num_replicas[i]

        self.current_time += seconds

    def optimize_policy(self):
        job_infos = self.get_job_infos()
        if job_infos:
            # check if active jobs change for shockwave
            if isinstance(self.policy, ShockWavePolicy):
                active_jobs = [job.name for job in self.jobs if job.submission_time <=
                               self.current_time and job.completion_time is None]
                if sorted(active_jobs) != sorted(self.active_jobs):
                    self.policy.resolve = True
                    self.policy.reestimate_share = True
                self.active_jobs = sorted(active_jobs)

            # Optimize allocations.
            node_infos = self.get_node_infos()
            self.vnode_allocations = {
                k: v for k, v in self.vnode_allocations.items() if k in job_infos}

            # run scheduling policy
            p_start_t = time.time()
            allocations, _ = self.policy.optimize(
                job_infos, node_infos, self.vnode_allocations, node_infos[0])
            repaired_allocations = self.repair_allocations(allocations)
            mapped_allocations = self.map_vnode_to_real_node(
                repaired_allocations)
            p_end_t = time.time()
            print(f"policy.optimize(): {round((p_end_t-p_start_t)*1000)}ms")

            if isinstance(self.policy, ShockWavePolicy):
                for jid in allocations:
                    assert len(allocations[jid]) > 0
                    if len(mapped_allocations[jid][1]) != len(allocations[jid]):
                        assert len(
                            mapped_allocations[jid][1]) == 0, f'{jid} shockwave alloc: {allocations[jid]} actual: {mapped_allocations[jid][1]}'
                        print(
                            f'!!! {jid} allocated by Shockwave but no allocation in hete cluster')
                        del self.policy._current_worker_assignments[jid]

            for job in self.jobs:
                if job.submission_time <= self.current_time and job.completion_time is None:
                    job.contention.append(len(job_infos))
                    # print(f"job: {job.name}, contention: {job.contention}")
                    cluster, alloc = mapped_allocations.get(
                        job.name, (None, ()))
                    _, prev_alloc = self.allocations.get(job.name, (None, ()))

                    if isinstance(self.policy, PolluxPolicyUnawareFix) or isinstance(self.policy, ShockWavePolicy):
                        assert len(alloc) == 0 or len(alloc) == job.target_num_replicas

                    # change in resources
                    if allocations.get(job.name) != self.allocations.get(job.name):
                        placement = []
                        for i in range(len(alloc)):
                            if i == 0 or alloc[i] != alloc[i - 1]:
                                placement.append(1)
                            else:
                                placement[-1] += 1
                        # fix placement if bad
                        nnodes = len(placement)
                        nreplicas = sum(placement)
                        if cluster is not None and nnodes > self.max_physical_nodes[cluster]:
                            n_reps_per_node = int(np.ceil(
                                nreplicas / self.max_physical_nodes[cluster]))
                            new_placement = []
                            while nreplicas > 0:
                                n_reps = min(n_reps_per_node, nreplicas)
                                new_placement.append(n_reps)
                                nreplicas -= n_reps
                            print(
                                f"FIXING PLACEMENT FOR LARGE CLUSTER: {cluster}, {placement} --> {new_placement}")
                            placement = new_placement
                        if job.current_cluster != cluster:
                            # migrated to another cluster
                            job.migrate(cluster, placement)
                        elif prev_alloc != alloc:
                            # reallocated within same cluster
                            job.reallocate(placement)
            # make a copy of allocations
            self.allocations = mapped_allocations
            self.vnode_allocations = repaired_allocations
        else:
            if isinstance(self.policy, ShockWavePolicy):
                self.policy.round_ptr += 1

    # input: vnode_allocs = dict(key=jobname, val=exact gpus allocated (virtual node IDs))
    # output: dict(key=jobname, val=(cluster_name, [cluster_gpu_ids]))
    def map_vnode_to_real_node(self, vnode_allocs):
        mapped_allocs = dict()
        for jobname, alloc in vnode_allocs.items():
            if alloc == [] or alloc is None:
                mapped_allocs[jobname] = (None, [])
            else:
                clusters = [self.vnode_cluster_node_id_map[v][0]
                            for v in alloc]
                assert len(
                    set(clusters)) == 1, f"failed to map alloc: {alloc}, clusters={set(clusters)}"
                mapped_allocs[jobname] = (clusters[0], sorted(
                    [self.vnode_cluster_node_id_map[v][1] for v in alloc]))
        return mapped_allocs

    def repair_allocations_mip(self, allocs):
        # convert allocs to cluster-specific allocs
        cluster_allocs = {}
        for jobname, alloc in allocs.items():
            if alloc:
                gpu_real_ids = [self.vnode_cluster_node_id_map[vnode_id]
                                for vnode_id in alloc]
                for cluster, node_id in gpu_real_ids:
                    cluster_allocs.setdefault(cluster, dict()).setdefault(
                        jobname, []).append(node_id)
        final_allocs = {}
        wasted_gpus = 0
        for jobname, alloc in allocs.items():
            if not alloc:
                final_allocs[jobname] = []
                continue
            # swap gpus
            gpu_ids = [self.vnode_cluster_node_id_map[vnode_id][0]
                       for vnode_id in alloc]
            gpu_counts = Counter(gpu_ids)
            unique_clusters = list(gpu_counts.keys())
            # cluster with highest GPU request is primary cluster
            primary_cluster = sorted(
                unique_clusters, key=lambda x: gpu_counts[x], reverse=True)[0]

            # TRY 1: remove other cluster gpus from alloc
            new_alloc = []
            for vnode_id in alloc:
                gpu_cluster, real_node_id = self.vnode_cluster_node_id_map[vnode_id]
                if gpu_cluster == primary_cluster:
                    new_alloc.append(vnode_id)
                else:
                    wasted_gpus += 1
            final_allocs[jobname] = new_alloc
        self.wasted_gpus.append(wasted_gpus)
        print(
            f"Avg. wasted GPUs: {np.mean(self.wasted_gpus)}, Current={wasted_gpus}")
        print(final_allocs)
        return final_allocs

    def repair_allocations_mip_fix(self, allocs):
        # convert allocs to cluster-specific allocs
        cluster_allocs = {}
        for jobname, alloc in allocs.items():
            if alloc:
                gpu_real_ids = [self.vnode_cluster_node_id_map[vnode_id]
                                for vnode_id in alloc]
                for cluster, node_id in gpu_real_ids:
                    cluster_allocs.setdefault(cluster, dict()).setdefault(
                        jobname, []).append(node_id)
        res_allocs = {}
        primary_cluster_dict = {cluster: [] for cluster in self.clusters}
        for jobname, alloc in allocs.items():
            if not alloc:
                print(f"!!! pollux gives no gpus to {jobname}")
                res_allocs[jobname] = []
                continue
            # swap gpus
            gpu_ids = [self.vnode_cluster_node_id_map[vnode_id][0]
                       for vnode_id in alloc]
            gpu_counts = Counter(gpu_ids)
            unique_clusters = list(gpu_counts.keys())
            # cluster with highest GPU request is primary cluster
            primary_cluster = sorted(
                unique_clusters, key=lambda x: gpu_counts[x], reverse=True)[0]
            primary_cluster_dict[primary_cluster].append(jobname)

        # force the actual gpus allocated == target_num_replicas
        print("### pollux alloc:", allocs)
        print("### prev alloc:", self.allocations)
        # print("### cluster allocs:", cluster_allocs)
        print("### primary cluster:", primary_cluster_dict)  # TODO: sort
        # print("### keep prev alloc:", res_allocs)

        for cluster, jobnames in primary_cluster_dict.items():
            # print("### \t allocate for", cluster)
            total_gpus = {vnode_id: 4 for vnode_id, cluster_node in self.vnode_cluster_node_id_map.items(
            ) if cluster_node[0] == cluster}
            free_gpus = collections.Counter(total_gpus)

            # target number of replicas dict
            job_target_num_replicas = {}
            for jobname in jobnames:
                if jobname not in res_allocs:
                    target_num_replicas = -1
                    for job in self.jobs:
                        if job.name == jobname:
                            target_num_replicas = job.target_num_replicas
                            break
                    assert target_num_replicas > 0
                    job_target_num_replicas[jobname] = target_num_replicas
            # sort jobs in decreaseing order of target number of replicas
            jobnames = sorted(
                jobnames, key=lambda x: job_target_num_replicas[x], reverse=True)

            for jobname in jobnames:
                if jobname not in res_allocs:
                    target_num_replicas = job_target_num_replicas[jobname]

                    if sum(list(free_gpus.values())) < target_num_replicas:
                        # no enough gpus for this job
                        print(f"!!! no enough {cluster} for {jobname}")
                        res_allocs[jobname] = []
                        continue
                    new_allocs = []
                    while len(new_allocs) < target_num_replicas:
                        node_idx, count = free_gpus.most_common(1)[0]
                        num = min(count, target_num_replicas - len(new_allocs))
                        new_allocs.extend([node_idx] * num)
                        free_gpus[node_idx] -= num

                    res_allocs[jobname] = new_allocs
                    assert len(res_allocs[jobname]) == target_num_replicas
        assert len(res_allocs) == len(allocs)
        print("### res allocs:", res_allocs)
        return res_allocs

    def repair_allocations_swap_idle_nodes(self, allocs):
        # create map of all gpus not allocated
        cluster_resources = {}
        ngpus_per_vnode = min(self.num_gpus.values())
        for vnode_id, (cluster, _) in self.vnode_cluster_node_id_map.items():
            cluster_resources.setdefault(cluster, dict())[vnode_id] = [
                vnode_id] * ngpus_per_vnode

        for jobname, alloc in allocs.items():
            for vgpu_id in alloc:
                # remove all gpus allocated to job
                cluster, _ = self.vnode_cluster_node_id_map[vgpu_id]
                cluster_resources[cluster][vgpu_id].pop(0)
        print(f"Free resources: {cluster_resources}")
        print(f"Allocs: {allocs}")

        # convert allocs to cluster-specific allocs
        cluster_allocs = {}
        for jobname, alloc in allocs.items():
            if alloc:
                gpu_real_ids = [self.vnode_cluster_node_id_map[vnode_id]
                                for vnode_id in alloc]
                for cluster, node_id in gpu_real_ids:
                    cluster_allocs.setdefault(cluster, dict()).setdefault(
                        jobname, []).append(node_id)

        final_allocs = {}
        wasted_gpus = dict()
        for jobname in allocs.keys():
            alloc = allocs[jobname]
            if not alloc:
                final_allocs[jobname] = []
                continue
            # get gpu cluster ids
            gpu_ids = [self.vnode_cluster_node_id_map[vnode_id][0]
                       for vnode_id in alloc]
            gpu_counts = Counter(gpu_ids)
            unique_clusters = list(gpu_counts.keys())

            # cluster with highest GPU request is primary cluster
            primary_cluster = sorted(
                unique_clusters, key=lambda x: gpu_counts[x], reverse=True)[0]

            # sort by max number of gpus idle in each node in primary cluster
            node_ngpus_idle = sorted([(k,  len(v)) for k, v in cluster_resources[primary_cluster].items(
            )], key=lambda x: x[1], reverse=True)

            # step-1: exchange gpus
            allocs_vnode_ids = Counter(alloc)  # indexed by vnode id
            new_allocs_vnode_ids = dict()
            for vnode_id in allocs_vnode_ids.keys():
                vnode_cluster = self.vnode_cluster_node_id_map[vnode_id][0]
                vnode_gpus = allocs_vnode_ids[vnode_id]

                # vnode_gpus are already in primary cluster
                if vnode_cluster == primary_cluster:
                    new_allocs_vnode_ids[vnode_id] = vnode_gpus
                else:
                    # replace vnode_gpus with exact node swap from idle gpus
                    vgpus_remaining_list = [
                        len(v) for v in cluster_resources[primary_cluster].values()]
                    if vnode_gpus in vgpus_remaining_list:
                        for p_vnode_id, p_idle_gpus in node_ngpus_idle:
                            if p_idle_gpus == vnode_gpus:
                                new_allocs_vnode_ids[p_vnode_id] = vnode_gpus
                                cluster_resources[primary_cluster][p_vnode_id] = [
                                ]
                                cluster_resources[vnode_cluster][vnode_id].extend(
                                    [vnode_id]*vnode_gpus)
                                print(
                                    f"REPAIR: SWAPPING: {vnode_id}<->{p_vnode_id}, ngpus={vnode_gpus}, {str().join(['#'] * 10)}")
                                break
                    else:
                        new_allocs_vnode_ids[vnode_id] = vnode_gpus
            alloc = []
            for vnode_id, vnode_gpus in new_allocs_vnode_ids.items():
                alloc.extend([vnode_id] * vnode_gpus)

            # step-2: trim non-primary cluster gpus from alloc
            new_alloc = []
            for vnode_id in alloc:
                gpu_cluster, real_node_id = self.vnode_cluster_node_id_map[vnode_id]
                if gpu_cluster == primary_cluster:
                    new_alloc.append(vnode_id)
                else:
                    wasted_gpus.setdefault(jobname, list()).append(
                        (gpu_cluster, vnode_id))
            final_allocs[jobname] = sorted(new_alloc)
        num_wasted_gpus = sum(len(v) for v in wasted_gpus.values())
        self.wasted_gpus.append(num_wasted_gpus)
        print(
            f"Avg. wasted GPUs: {np.mean(self.wasted_gpus)}, Current={wasted_gpus}")
        return final_allocs

    # fixes allocations by keeping GPUs making up majority of allocations (dropping others)
    # prio_order: ordering of GPU types to keep if equal probability of dropping
    # returns: new_allocs --> fixed allocs
    #          drop_count --> number of GPUs dropped for each job
    def fix_allocations_by_dropping(self, allocs, prio_order=None):
        if prio_order is None:
            prio_order = {"dgx-ext": 0, "azure": 1,
                          "quad": 2, "rtx": 3, "aws": 4}
        print(allocs)
        # convert allocs to cluster-specific allocs
        new_allocs = dict()
        drop_count = dict()
        for k, v in allocs.items():
            if v is None or len(v) == 0:
                new_allocs[k] = v
                drop_count[k] = 0
                print(f"no alloc for {k}")
                continue
            # count unique GPUs of each type
            gpu_real_ids = [self.vnode_cluster_node_id_map[vnode_id]
                            for vnode_id in v]
            gpu_types = [x[0] for x in gpu_real_ids]
            gpu_counts = Counter(gpu_types).most_common()

            # no heterogeneous allocs
            if len(set(gpu_counts)) == 1:
                new_allocs[k] = v
                drop_count[k] = 0
                print(f"no het alloc for {k}")
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
            print(f"Choosing GPU type: {chosen_type}")
            new_allocs[k] = []
            for v2 in v:
                gpu_t, node_id = self.vnode_cluster_node_id_map[v2]
                if gpu_t == chosen_type:
                    new_allocs[k].append(v2)
                else:
                    print(
                        f"Dropping replica on vnode-{v2} [{gpu_t},node-{node_id}] for {k}")
        # print(f"New Allocs: {new_allocs}, drop count = {drop_count}")
        return new_allocs

    def repair_allocations(self, allocs):
        if isinstance(self.policy, SiaUnaware):
            return self.repair_allocations_mip(allocs)
        elif isinstance(self.policy, PolluxPolicyUnaware):
            # return self.repair_allocations_mip(allocs)
            return self.fix_allocations_by_dropping(allocs)
        elif isinstance(self.policy, PolluxPolicyUnawareFix) or isinstance(self.policy, ShockWavePolicy):
            return self.repair_allocations_mip_fix(allocs)
        else:
            assert False, "no alloc repair func for policy={self.policy.__name__}"

    def step_log(self):
        step_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "cur_step": int(self.get_shockwave_job_info(job).cur_step) if isinstance(self.policy, ShockWavePolicy) else -1,
                    "total_steps": int(self.get_shockwave_job_info(job).total_steps) if isinstance(self.policy, ShockWavePolicy) else -1,
                    "num_restarts": job.num_restarts,
                    "cluster": job.current_cluster,
                    "allocation": self.allocations.get(job.name, tuple()),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * int(sum(job.placement)),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "start_time": job.start_time,
                    "completion_time": job.completion_time,
                    "n_avg": np.mean(np.asarray(job.contention))
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
            "wasted_gpus": float(self.wasted_gpus[-1]) if len(self.wasted_gpus) > 0 else 0
        }
        self.logs.append(step_log)

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, PolluxPolicyUnawareFix):
                    job_infos[job.name] = self.get_unaware_job_info_fix(job)
                elif isinstance(self.policy, ShockWavePolicy):
                    job_infos[job.name] = self.get_shockwave_job_info(job)
                else:
                    job_infos[job.name] = self.get_unaware_job_info(job)
        return job_infos

    def get_unaware_job_info(self, job):
        job_app = job.applications["aws"]
        max_replicas = min(
            max(2 * job.max_profiled_replicas(), 1),
            self.clusterwide_max_replicas,
            job.applications["aws"].max_batch_size // job_app.min_local_bsz)
        job_info = JobInfoUnaware(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=max_replicas,
            preemptible=True,
        )

        if job_app.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_unaware_job_info_fix(self, job):
        job_app = job.applications["aws"]
        target_replicas = min(
            job.target_num_replicas,
            self.clusterwide_max_replicas,
            job.applications["aws"].max_batch_size // job_app.min_local_bsz)
        job_info = JobInfoUnaware(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=target_replicas,
            max_replicas=target_replicas,
            preemptible=True,
        )

        if job_app.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_shockwave_job_info(self, job):
        job_app = job.applications["aws"]
        job_info = JobInfoUnaware(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=job.target_num_replicas,
            max_replicas=job.target_num_replicas,
            preemptible=True,
        )
        if job_app.name == "ncf":
            job_info.max_replicas = 1
        job_info.applications = job.applications

        # shockwave add-on
        job_info.epochs = job.completion_epoch
        job_info.epoch_progress = job.epoch
        job_info.epoch_duration = job.epoch_duration
        job_info.target_batch_size = job.target_batch_size
        job_info.scale_factor = job.target_num_replicas
        job_info.age = self.current_time - job.submission_time
        job_info.submission_time = job.submission_time
        job_info.execution_time = job.execution_time
        # print(f'{job_app.name} bsz: {job.target_batch_size} epoch: {job.epoch} progress: {job.progress} epoch: {job.epoch} completion_epoch: {job.completion_epoch}')
        if job.completion_time is None:
            job_info.cur_step = job_app.get_cur_iteration(
                job.target_batch_size, job.epoch, job.progress)
            job_info.total_steps = job_app.get_iteration(
                job.target_batch_size, job.completion_epoch)
        else:
            job_info.cur_step = job_app.get_iteration(
                job.target_batch_size, job.completion_epoch)
            job_info.total_steps = job_app.get_iteration(
                job.target_batch_size, job.completion_epoch)
        job_info.slowdown_factor = job.random_slowdown_factor
        assert job_info.cur_step >= 0 and job_info.cur_step <= job_info.total_steps
        return job_info

    def get_node_infos(self, num_nodes=None):
        node_infos = {idx: NodeInfo({"nvidia.com/gpu": 4}, preemptible=False)
                      for idx in range(len(self.vnode_cluster_node_id_map))}
        return node_infos

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        if len(self.logs) > 0:
            return {
                val["name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None
            }
        else:
            return {}
