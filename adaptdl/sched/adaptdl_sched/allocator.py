# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
import dateutil.parser
import kubernetes_asyncio as kubernetes
import logging
import time
from datetime import datetime, timezone

from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl.sched_hints import PERF_PARAMS
from adaptdl_sched.policy.applications import APPLICATIONS
from adaptdl_sched.policy.applications_v2 import APPLICATIONS as APPLICATIONS_V2
from adaptdl_sched.policy.optimus import OptimusPolicy
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.unaware_pollux import UnawarePolluxPolicy
from adaptdl_sched.policy.gavel import GavelPolicy

from adaptdl_sched.policy.mip import MIPPolicy
from adaptdl_sched.policy.tiresias import TiresiasPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
from adaptdl_sched.resources import (get_node_unrequested, get_pod_requests,
                                     set_default_resources)
from adaptdl_sched.utils import patch_job_status
from adaptdl_sched.cluster_expander import ClusterExpander
from adaptdl_sched.config import allowed_taints

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


POLICY = "unaware_pollux"
assert POLICY in ["optimus", "pollux", "unaware_pollux", "tiresias", "mip", "gavel"]
PHOEBE_DEBUG = True

class AdaptDLAllocator(object):
    # use new APPLICATIONS with support for multiple GPU types
    _use_applications_v2 = True
    # default cluster to use for choosing APPLICATIONS
    _default_cluster = "rtx"
    # delay between consecutive invocations of the scheduler loop
    _sched_trigger_interval = 60

    def __init__(self, expander):
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._cluster_expander = expander
        if POLICY == "pollux":
            self._policy = PolluxPolicy()
        elif POLICY == "unaware_pollux":
            self._policy = UnawarePolluxPolicy()
        elif POLICY == "gavel":
            self._policy = GavelPolicy(interval=360)
            self._sched_trigger_interval = 360
        elif POLICY == "optimus":
            self._policy = OptimusPolicy()
        elif POLICY == "tiresias":
            self._policy = TiresiasPolicy(time.time)
        elif POLICY == "mip":
            self._policy = MIPPolicy()

    async def run(self):
        while True:
            LOG.info("Running allocator loop")
            nodes, node_template = await self._find_nodes()
            LOG.info("Node resources: %s",
                     {k: v.resources for k, v in nodes.items()})
            jobs, prev_allocations = await self._find_jobs_and_allocations()
            LOG.info("Job resources: %s",
                     {k: v.resources for k, v in jobs.items()})
            start = time.time()
            allocations = self._allocate(jobs, nodes, prev_allocations,
                                         node_template)
            duration = time.time() - start
            LOG.info("Allocations (in %.3f sec): %s", duration, allocations)
            await self._update_allocations(allocations)
            LOG.info(f"Sleep for {self._sched_trigger_interval} seconds")
            await asyncio.sleep(self._sched_trigger_interval)

    async def _update_allocations(self, allocations):
        job_list = await self._objs_api.list_namespaced_custom_object(
            "adaptdl.petuum.com", "v1", "", "adaptdljobs")
        for job in job_list["items"]:
            namespace = job["metadata"]["namespace"]
            name = job["metadata"]["name"]
            job_allocation = job.get("status", {}).get("allocation", [])
            new_allocation = list(allocations.get((namespace, name), []))
            if list(job_allocation) != new_allocation:
                patch = {"status": {"allocation": new_allocation}}
                LOG.info("Patch AdaptDLJob %s/%s: %s", namespace, name, patch)
                await patch_job_status(self._objs_api, namespace, name, patch)

    async def _find_nodes(self):
        node_infos = {}
        node_list = await self._core_api.list_node()
        # Find all non-AdaptDL pods which are taking up resources and subtract
        # those resources from the available pool. Apparently there's not a
        # more efficient way to get currently available resources in k8s?. We
        # also check if we have reached the pod limit on the node. This number
        # denotes (allocatable pods - Non-terminated pods) on that node.
        pod_list = await self._core_api.list_pod_for_all_namespaces(
                label_selector="!adaptdl/job")
        for node in node_list.items:
            if allowed_taints(node.spec.taints):
                resources = get_node_unrequested(node, pod_list.items)
                if not resources.get("pods"):
                    LOG.warning(f"node {node.metadata.name} "
                                "has no free pods available.")
                node_infos[node.metadata.name] = NodeInfo(resources, False)
        # For cluster autoscaling: to determine if additional nodes would be
        # helpful, add a few "virtual" nodes which only become available in
        # "eta" seconds. Currently, we only consider as many virtual nodes as
        # there are real nodes. We infer each resource to be the maximum amount
        # observed in any real node.
        max_resources = {}
        for node_name in node_infos:
            for key, val in node_infos[node_name].resources.items():
                if key not in max_resources or val > max_resources[key]:
                    max_resources[key] = val
        node_template = NodeInfo(max_resources, True)
        return node_infos, node_template

    async def _find_jobs_and_allocations(self):
        LOG.info(f"Finding existing jobs and allocations")
        job_list = await self._objs_api.list_namespaced_custom_object(
            "adaptdl.petuum.com", "v1", "", "adaptdljobs")
        job_infos = {}
        allocations = {}
        current_ts = datetime.now(timezone.utc)
        for job in job_list["items"]:
            if job.get("status", {}).get("phase") \
                    not in ["Pending", "Running", "Starting", "Stopping"]:
                continue
            if "placement" in job["spec"]:
                continue
            if "allocation" in job.get("status", {}):
                namespace = job["metadata"]["namespace"]
                name = job["metadata"]["name"]
                allocations[namespace, name] = \
                    list(job["status"]["allocation"])
            job["spec"]["template"]["spec"] = \
                set_default_resources(job["spec"]["template"]["spec"])
            resources = get_pod_requests(job["spec"]["template"]["spec"])
            hints = job.get("status", {}).get("train", {})
            max_replicas = max(2 * hints.get("maxProfiledReplicas", 0), 1)
            if job["spec"].get("maxReplicas"):
                max_replicas = min(max_replicas, job["spec"]["maxReplicas"])
            min_replicas = job["spec"].get("minReplicas", 0)
            # max_replicas should be greater or equal to min_replicas
            max_replicas = max(max_replicas, min_replicas)
            preemptible = job["spec"].get("preemptible", True)
            if {"perfParams", "initBatchSize"} <= hints.keys() and preemptible:
                max_batch_size = (hints.get("maxBatchSize")
                                  or hints["initBatchSize"])
                if hints.get("localBszBounds"):
                    min_local_bsz = hints["localBszBounds"][0] or 1
                    # Make sure max_batch_size / replicas >= min_local_bsz
                    if max_batch_size < min_local_bsz * max_replicas:
                        max_replicas = int(max_batch_size / min_local_bsz)
                if "gradParams" in hints:
                    grad_params = GradParams(hints["gradParams"]["norm"], hints["gradParams"]["var"])
                else:
                    grad_params = GradParams(0.0, 1.0)
                
                # process as heterogeneity-oblivious
                if POLICY != "mip":
                    perf_params = PerfParams(*[hints["perfParams"][k]
                                            for k in PERF_PARAMS.keys()])
                    goodput_fn = GoodputFunction(perf_params, grad_params,
                                                hints["initBatchSize"])
                    speedup_fn = SpeedupFunction(goodput_fn, hints.get("maxBatchSize"), hints.get("localBszBounds"),
                                                 hints.get("gradientAccumulation", False))
                else:
                    # process as heterogeneity-aware
                    perf_params_dict = hints["perfParamsDict"]
                    seed_perf_params_dict = hints.get("perfParamsHintDict", dict())
                    LOG.info(f"Perf params dict: {perf_params_dict}")
                    LOG.info(f"Seed perf params dict: {seed_perf_params_dict}")
                    perf_params = dict()
                    for gpu_type, gpu_perf_params in perf_params_dict.items():
                        perf_params[gpu_type] = PerfParams(*[gpu_perf_params[k]
                                            for k in PERF_PARAMS.keys()])
                    for gpu_type, gpu_perf_params in seed_perf_params_dict.items():
                        if gpu_type not in perf_params:
                            LOG.info(f"Using seeded profile for GPU type: {gpu_type}")
                            perf_params[gpu_type] = PerfParams(*[gpu_perf_params[k] for k in PERF_PARAMS.keys()])

                    speedup_fn = dict()
                    local_bsz_bounds = hints.get("localBszBounds", None)
                    local_bsz_bounds_dict = hints.get("localBszBoundsDict", dict())
                    LOG.info(f"Local Bsz Bounds: {local_bsz_bounds}, local bsz bounds dict: {local_bsz_bounds_dict}")
                    for gpu_type, gpu_perf_params in perf_params.items():
                        goodput_fn = GoodputFunction(gpu_perf_params, grad_params,
                                                     hints["initBatchSize"])
                        gpu_bsz_bounds = local_bsz_bounds_dict.get(gpu_type, local_bsz_bounds)
                        speedup_fn[gpu_type] = SpeedupFunction(goodput_fn, hints.get("maxBatchSize"), 
                                                               gpu_bsz_bounds,hints.get("gradientAccumulation", False))
            else:
                # baseline speedup fns
                speedup_fn = lambda n, r: r  # noqa: E731
            creation_ts = dateutil.parser.isoparse(
                    job["metadata"]["creationTimestamp"])
            attained_service = job.get("status", {}).get("attainedService", 0)
            attained_service_ts = job.get("status", {}).get("attainedServiceTimestamp")
            if attained_service_ts:
                attained_service_ts = dateutil.parser.isoparse(attained_service_ts)
                duration = (current_ts - attained_service_ts).total_seconds()
                attained_service += duration * job.get("status", {}).get("replicas", 0)
            namespace = job["metadata"]["namespace"]
            name = job["metadata"]["name"]
            job_info = JobInfo(resources, speedup_fn, creation_ts,
                               min_replicas, max_replicas, preemptible)
            job_info.attained_service = attained_service
            job_info.epoch = job.get("status", {}).get("train", {}).get("epoch", 0)
            if self._use_applications_v2:
                chosen_cluster = self._default_cluster or "rtx"
                job_info.application = APPLICATIONS_V2[chosen_cluster][job["spec"]["application"]]
            else:
                job_info.application = APPLICATIONS[job["spec"]["application"]]
            job_info.target_num_replicas = int(job["spec"]["targetNumReplicas"])
            job_info.target_batch_size = int(job["spec"]["targetBatchSize"])
            job_info.num_restarts = job.get("status", {}).get("group") or 0
            job_info.age = (current_ts - creation_ts).total_seconds()
            if POLICY == "optimus":
                job_info.max_replicas = int(job_info.target_batch_size / job_info.application.min_local_bsz)
                if job["spec"].get("maxReplicas"):
                    job_info.max_replicas = min(job_info.max_replicas, job["spec"]["maxReplicas"])
            elif POLICY == "tiresias":
                job_info.max_replicas = job_info.target_num_replicas
            elif POLICY == "gavel":
                job_info.min_replicas = job_info.max_replicas = job_info.target_num_replicas
            job_infos[(namespace, name)] = job_info
        return job_infos, allocations

    def _allocate(self, jobs, nodes, prev_allocations, node_template):
        for job_key in list(jobs):
            job_resources = jobs[job_key].resources
            for node in nodes.values():
                if all(val <= node.resources.get(key, 0)
                       for key, val in job_resources.items()):
                    # Found a node which can fit a replica of this job.
                    break
            else:
                # No node can fit a replica of this job.
                # TODO: propagate this to the controller so the job is Failed.
                LOG.warning("Job %s cannot be scheduled!", job_key)
                # TODO: (suhasj) -- uncomment the pop operation after Phoebe experiments
                if not PHOEBE_DEBUG:
                    jobs.pop(job_key)
        allocations = {}
        if not jobs:
            # There are no jobs, let the expander shrink the cluster.
            self._cluster_expander.fit([])
        elif jobs and nodes:
            allocations, desired_nodes = self._policy.optimize(
                jobs, nodes, prev_allocations, node_template)
            if desired_nodes < len(nodes):
                active_nodes = list(set.union(*map(set, allocations.values())))
            else:
                active_nodes = list(nodes)
                while len(active_nodes) < desired_nodes:
                    active_nodes.append(f"vnode-~{desired_nodes-len(active_nodes)}")
            # self._cluster_expander.fit(active_nodes)
            LOG.info("Active nodes: %s", active_nodes)
        elif jobs:
            # Expand job ASG from zero nodes.
            # Assumption is AdaptDL is running on a different ASG
            self._cluster_expander.fit(['~1'])
        return allocations


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    expander = ClusterExpander()
    allocator = AdaptDLAllocator(expander)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(asyncio.gather(
    #    expander.run(),
        allocator.run(),
    ))
    loop.close()
