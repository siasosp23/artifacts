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


import collections
import json
import pickle
import time
import logging

import numpy as np

import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.goodput import GoodputFunction, fit_perf_params
from adaptdl.sched_hints import SCHED_HINTS, PERF_PARAMS, NODE_TO_CLUSTER_MAP, post_sched_hints


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

def report_train_metrics(epoch, loss, **kwargs):
    if adaptdl.env.replica_rank() > 0:
        return
    with open(adaptdl.env.checkpoint_path() + "/train.txt", "a") as f:
        json.dump(dict(
            time=time.time(),
            progress=get_progress(),
            epoch=epoch,
            loss=loss,
            **kwargs
        ), f)
        f.write("\n")


def report_valid_metrics(epoch, loss, **kwargs):
    if adaptdl.env.replica_rank() > 0:
        return
    with open(adaptdl.env.checkpoint_path() + "/valid.txt", "a") as f:
        json.dump(dict(
            time=time.time(),
            progress=get_progress(),
            epoch=epoch,
            loss=loss,
            **kwargs
        ), f)
        f.write("\n")


def profile_step_start(atomic_bsz):
    state = _metrics_state()
    state.atomic_bsz = atomic_bsz
    state.step_start = time.time()
    state.sync_time = 0.0


def profile_sync_time(sync_time):
    _metrics_state().sync_time += sync_time


_PREV_REPORT = None

def profile_step_commit(epoch, batch_size, accumulation_step=False):
    global _PREV_REPORT
    state = _metrics_state()
    step_time = time.time() - state.step_start
    num_nodes = adaptdl.env.num_nodes()
    num_replicas = adaptdl.env.num_replicas()
    key = (num_nodes, num_replicas, state.atomic_bsz)

    # create new table for new GPU type profiles
    gpu_type = adaptdl.env.gpu_type()
    if gpu_type not in state.profile_dict:
        print(f"Adding GPU type: {gpu_type}")
        state.profile_dict[gpu_type] = collections.defaultdict(collections.Counter)

    if accumulation_step:
        state.profile[key]["accum_step_time"] += step_time
        state.profile[key]["accum_count"] += 1
        state.profile_dict[gpu_type][key]["accum_step_time"] += step_time
        state.profile_dict[gpu_type][key]["accum_count"] += 1
    else:
        state.profile[key]["optim_step_time"] += step_time
        state.profile[key]["optim_sync_time"] += state.sync_time
        state.profile[key]["optim_count"] += 1
        state.profile_dict[gpu_type][key]["optim_step_time"] += step_time
        state.profile_dict[gpu_type][key]["optim_sync_time"] += state.sync_time
        state.profile_dict[gpu_type][key]["optim_count"] += 1
    del state.atomic_bsz
    del state.step_start
    del state.sync_time
    if not accumulation_step:
        if _PREV_REPORT is None:
            _PREV_REPORT = time.time()
        if adaptdl.env.replica_rank() == 0 and time.time() - _PREV_REPORT > 30:
            _fit_perf_params()
            _report_sched_hints(epoch, batch_size)
            _PREV_REPORT = time.time()


_GRAD_PARAM_DICT = {}


def update_grad_params(edp_key, grad_norm_sqr, grad_variance):
    global _GRAD_PARAM_DICT
    _GRAD_PARAM_DICT[edp_key] = np.asarray([grad_norm_sqr, grad_variance])
    grad_params = sum(_GRAD_PARAM_DICT.values())
    _metrics_state().grad_params = (grad_params[0], grad_params[1])


def update_progress(progress):
    _metrics_state().progress = progress


def get_progress():
    return _metrics_state().progress

def set_batch_size(init_batch_size, max_batch_size, local_bsz_bounds,
                   gradient_accumulation, local_bsz_bounds_dict = None):
    state = _metrics_state()
    state.init_batch_size = init_batch_size
    state.max_batch_size = max_batch_size
    state.local_bsz_bounds = local_bsz_bounds
    state.gradient_accumulation = gradient_accumulation
    # additional flexibility to set varied local bsz bounds per GPU type
    if local_bsz_bounds_dict is not None:
        state.local_bsz_bounds_dict = local_bsz_bounds_dict

def get_goodput_fn(gpu_type=None):
    state = _metrics_state()
    if state.grad_params is None or state.perf_params is None:
        return None
    if gpu_type is None:
        return GoodputFunction(state.perf_params, state.grad_params, state.init_batch_size)
    else:
        if gpu_type in state.perf_params_dict:
            return GoodputFunction(state.perf_params_dict[gpu_type], state.grad_params, 
                                   state.init_batch_size)
        elif gpu_type in state.seed_perf_params_dict:
            print(f"Using seed throughput function")
            return GoodputFunction(state.seed_perf_params_dict[gpu_type], state.grad_params, 
                                   state.init_batch_size)
        else:
            print(f"couldn't find perf params for {gpu_type}")
            return None

def _fit_perf_params_helper(profile):
    # Convert profile into numpy arrays.
    num_nodes, num_replicas, atomic_bsz = (np.array(k) for k in zip(*profile))
    accum_step_time = np.array([v.get("accum_step_time", 0.0)
                                for v in profile.values()])
    accum_count = np.array([v.get("accum_count", 0) for v in profile.values()])
    optim_step_time = np.array([v.get("optim_step_time", 0.0)
                                for v in profile.values()])
    optim_sync_time = np.array([v.get("optim_sync_time", 0.0)
                                for v in profile.values()])
    optim_count = np.array([v.get("optim_count", 0) for v in profile.values()])
    assert np.all(optim_count > 0)
    # Non-sync time during optimization steps should be approximately equal to
    # accumulation step time, combine those data points.
    assert np.all(optim_step_time >= optim_sync_time)
    accum_step_time += optim_step_time - optim_sync_time
    accum_count += optim_count
    accum_step_time /= accum_count
    optim_step_time /= optim_count
    return fit_perf_params(num_nodes, num_replicas, atomic_bsz, accum_step_time, optim_step_time)

def _fit_perf_params():
    state = _metrics_state()
    # fit perf params per GPU type
    for gpu_type, gpu_profile in state.profile_dict.items():
        print(f"Fitting perf params for gpu type: {gpu_type}")
        profile = {k: v for k, v in gpu_profile.items() if v.get("optim_count")}
        state.perf_params_dict[gpu_type] = _fit_perf_params_helper(profile)
    # fit perf params mixed (for pollux and backward compatibility)
    profile = {k: v for k, v in state.profile.items() if v.get("optim_count")}
    state.perf_params = _fit_perf_params_helper(profile)

# scheduling hints for this job (set by master)
_SEED_SCHED_HINT_DICT = None

def _report_sched_hints(epoch, batch_size):
    assert adaptdl.env.replica_rank() == 0
    state = _metrics_state()
    # Scheduling hints
    sched_hints = SCHED_HINTS.copy()
    sched_hints["perfParams"] = {k: v for (k, v) in
                                 zip(PERF_PARAMS.keys(),
                                 state.perf_params)}
    sched_hints["maxBatchSize"] = state.max_batch_size
    sched_hints["localBszBounds"] = state.local_bsz_bounds
    sched_hints["initBatchSize"] = state.init_batch_size
    if state.grad_params:
        sched_hints["gradParams"] = {}
        sched_hints["gradParams"]["norm"] = state.grad_params[0]
        sched_hints["gradParams"]["var"] = state.grad_params[1]
    sched_hints["maxProfiledReplicas"] = max(key[1] for key in state.profile)
    sched_hints["gradientAccumulation"] = state.gradient_accumulation
    sched_hints["epoch"] = epoch
    sched_hints["batchSize"] = batch_size
    sched_hints["localBszBoundsDict"] = {k : v for (k, v) in state.local_bsz_bounds_dict.items()}
    # { gpu_type -> gpu_perf_params }
    perf_params_dict = dict()
    for gpu_type, gpu_perf_params in state.perf_params_dict.items():
        perf_params_dict[gpu_type] = {k: v for (k, v) in zip(PERF_PARAMS.keys(), gpu_perf_params)}
    sched_hints["perfParamsDict"] = perf_params_dict
    sched_hints["perfParamsHintDict"] = _SEED_SCHED_HINT_DICT
    if not sched_hints["perfParamsHintDict"]:
        print(f"Did not get seed sched hints")
    post_sched_hints(sched_hints, adaptdl.env.job_id())

# profiles_dicts: list(profiles)
# profile: { profile_key -> profile_value }
def seed_sched_hints(profiles):
    if adaptdl.env.replica_rank() != 0:
        return
    
    # already computed seed params
    global _SEED_SCHED_HINT_DICT
    if _SEED_SCHED_HINT_DICT:
        return

    seed_profiles_dict = dict()
    seed_params_dict = dict()
    for profile in profiles:
        gpu_type = profile['gpu_type']
        if gpu_type not in seed_profiles_dict:
            seed_profiles_dict[gpu_type] = dict()
        # Convert profile into numpy arrays.
        placement = profile['placement']
        nnodes = len(str(placement))
        nreplicas = sum([int(v) for v in str(placement)])
        seed_profiles_dict[gpu_type].setdefault('num_nodes', list()).append(nnodes)
        seed_profiles_dict[gpu_type].setdefault('num_replicas', list()).append(nreplicas)
        seed_profiles_dict[gpu_type].setdefault('local_bsz', list()).append(int(profile['local_bsz']))
        seed_profiles_dict[gpu_type].setdefault('step_time', list()).append(float(profile['step_time']))
        seed_profiles_dict[gpu_type].setdefault('sync_time', list()).append(float(profile['sync_time']))

    for gpu_type, gpu_profile in seed_profiles_dict.items():
        num_nodes = np.array(gpu_profile['num_nodes'])
        num_replicas = np.array(gpu_profile['num_replicas'])
        local_bsz = np.array(gpu_profile['local_bsz'])
        step_time = np.array(gpu_profile['step_time'])
        sync_time = np.array(gpu_profile['sync_time'])
        LOG.info(f"Seed profile: {gpu_type} -> {num_nodes}, {num_replicas}, {local_bsz}, {step_time}, {sync_time}")

        compute_time = step_time - sync_time
        perf_params = fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time)
        seed_params_dict[gpu_type] = {k: v for (k, v) in zip(PERF_PARAMS.keys(), perf_params)}
        print(f"Seed perf params: {gpu_type} -> {perf_params}")
    # set seed hints
    _SEED_SCHED_HINT_DICT = seed_params_dict
    LOG.info(f"_SEED_SCHED_HINT_DICT :{_SEED_SCHED_HINT_DICT}")


class _MetricsState(adaptdl.checkpoint.State):
    def __init__(self):
        super().__init__("adaptdl-metrics")
        self.profile = collections.defaultdict(collections.Counter)
        self.perf_params = None
        self.grad_params = None
        self.init_batch_size = None
        self.max_batch_size = None
        self.local_bsz_bounds = None
        self.gradient_accumulation = False
        self.progress = 0.0  # Progress in scale-invariant iterations.
        # heterogeneity-aware state
        self.gpu_type = None
        self.profile_dict = dict()
        self.local_bsz_bounds_dict = dict()
        self.perf_params_dict = dict()

        # seed perf params
        self.do_seed_perf_params = False
        self.seed_perf_params_dict = dict()

    def save(self, fileobj):
        pickle.dump(self.profile, fileobj)
        pickle.dump(self.perf_params, fileobj)
        pickle.dump(self.grad_params, fileobj)
        pickle.dump(self.init_batch_size, fileobj)
        pickle.dump(self.max_batch_size, fileobj)
        pickle.dump(self.local_bsz_bounds, fileobj)
        pickle.dump(self.gradient_accumulation, fileobj)
        pickle.dump(self.progress, fileobj)
        pickle.dump(self.gpu_type, fileobj)
        pickle.dump(self.profile_dict, fileobj)
        pickle.dump(self.local_bsz_bounds_dict, fileobj)
        pickle.dump(self.perf_params_dict, fileobj)

    def load(self, fileobj):
        self.profile = pickle.load(fileobj)
        self.perf_params = pickle.load(fileobj)
        self.grad_params = pickle.load(fileobj)
        self.init_batch_size = pickle.load(fileobj)
        self.max_batch_size = pickle.load(fileobj)
        self.local_bsz_bounds = pickle.load(fileobj)
        self.gradient_accumulation = pickle.load(fileobj)
        self.progress = pickle.load(fileobj)
        self.gpu_type = pickle.load(fileobj)
        self.profile_dict = pickle.load(fileobj)
        self.local_bsz_bounds_dict = pickle.load(fileobj)
        self.perf_params_dict = pickle.load(fileobj)

def _metrics_state():
    global _METRICS_STATE
    if _METRICS_STATE is None:
        _METRICS_STATE = _MetricsState()
        adaptdl.checkpoint.load_state(_METRICS_STATE)
        # set GPU type to current GPU type
        pod_gpu_type = adaptdl.env.gpu_type()
        if pod_gpu_type not in NODE_TO_CLUSTER_MAP.values():
            print(f"Error -- invalid gpu type : {pod_gpu_type}")
        else:
            print(f"Initialized GPU type --> {pod_gpu_type}")
            _METRICS_STATE.gpu_type = pod_gpu_type
    return _METRICS_STATE


_METRICS_STATE = None
