import math
import time
import numpy as np
import random

from applications import APPLICATIONS
import simulator_config as sim_config
from simulator_config import SLOWDOWNS
from goodput import GoodputFunction, GoodputFunctionPMP, fit_perf_params, PerfParams, GradParams
from speedup import SpeedupFunction, UncachedSpeedupFunction
from utils import JobInfo, NodeInfo

MAX_SLOWDOWN = 15

class Job(object):
    def __init__(self, name, applications, submission_time,
                 target_num_replicas=None, target_batch_size=None,
                 cache_speedups=False, h_unaware=False, category=None):
        ## Job attributes
        self.name = name
        self.app_name = list(applications.values())[0].name
        # applications is a dict w/ key=cluster_name, val=cluster_specific_App
        self.applications = applications
        self.category = category
        
        if self.category == "rigid":
            self.cluster_throughputs = None
        
        # submission_time: when it was submitted
        self.submission_time = submission_time
        # target_num_replicas: requested num replicas
        self.target_num_replicas = target_num_replicas
        # target_batch_size: requested batch size
        self.target_batch_size = target_batch_size
        # switch to control bsz tuning
        self.enable_bsz_tuning = True

        # switch to control if job is heterogeneity-aware
        self.h_unaware = h_unaware
        # profile is also a dict w/ key=cluster_name, val=cluster_specific_profile
        # if h_unaware is True, profiles is just all profiles merged into one dict
        self.profiles = dict()
        # perf_params is also a dict w/ key=cluster_name, val=cluster_specific_perf_params
        if self.h_unaware:
            self.perf_params = None
        else:
            self.perf_params = dict()
            for cluster in self.applications.keys():
                self.profiles[cluster] = dict()
                self.perf_params[cluster] = None
        # Optimization state
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.reference_app = self.applications[sim_config.calibrate_cluster]
        
        ## Job state
        # start_time: when it started running
        self.start_time = None
        # completion_time: when it finished running
        self.completion_time = None
        # current_time: current wall-clock time
        self.current_time = 0
        # (overhead) time to checkpoint + restore job
        # >0 if an operation is pending/executing, 0 otherwise
        self.rescale_time = 0
        self.max_rescale_time = self.reference_app.rescale_time
        # N-tuple: (x_1, x_2, ..., x_N) where x_i is the number of replicas in node i
        self.placement = ()
        # GPU-seconds allocated
        self.attained_service = 0
        # GPU-seconds used by job
        self.used_gpu_seconds = 0
        # GPU-seconds wasted by job
        self.wasted_gpu_seconds = 0

        # number of jobs contending for resources (one value per round)
        self.contention = []
        # number of times the job has been restarted
        self.num_restarts = None
        # number of times the job has been migrated between different clusters
        # note: num_migrations <= num_restarts
        self.num_migrations = None
        # events corresponding to job state changes
        self.bsz_update_events = []
        self.rescale_events = []
        self.migrate_events = []
        # num_stages > 1 for Pipeline-Model-Parallelism(PMP) jobs
        self.num_stages = self.reference_app.num_stages

        # cluster that this job is allocated to
        self.current_cluster = None
        self.speedup_fn_class = SpeedupFunction if cache_speedups else UncachedSpeedupFunction

        # shockwave addons
        self.execution_time = 0
        self.epoch_duration = []
        self.completion_epoch = self.reference_app.get_completion_epoch(
            self.target_batch_size) if self.target_batch_size is not None else None

        # calibration factor for this job (calibration step to account for real-world runs being slower than simulator)
        self.calibration_factor = 1 / SLOWDOWNS[self.app_name]
        if sim_config.log_cluster_verbose: 
            print(f"{self.name}: SpeedupFnClass={self.speedup_fn_class.__name__}, calibration-factor={self.calibration_factor:.2f}")
    
    # optimizes perf params given an input perf profile
    # perf_profile is a dict w/ key=(num_nodes, num_replicas, local_bsz), val=(step_time, sync_time)
    # returns a PerfParams object
    def optimize_perf_params(self, perf_profile):
        if perf_profile is None:
            return None
        num_nodes = np.array([key[0] for key in perf_profile])
        num_replicas = np.array([key[1] for key in perf_profile])
        local_bsz = np.array([key[2] for key in perf_profile])
        step_time = np.array([val[0] for val in perf_profile.values()])
        sync_time = np.array([val[1] for val in perf_profile.values()])
        compute_time = step_time - sync_time

        perf_params = fit_perf_params(
                num_nodes, num_replicas, local_bsz, compute_time, step_time)
        return perf_params

    def seed_profiles(self, max_num_nodes, max_num_replicas):
        print(f"Seeding profiles for job: {self.name}")
        for cluster, cluster_app in self.applications.items():
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]

            # add placements data
            if max_num_nodes > 0:
                placements_selector = (cluster_app.placements.num_nodes <= max_num_nodes) & (
                    cluster_app.placements.num_replicas <= max_num_replicas)
                df = cluster_app.placements[placements_selector]
            else:
                df = cluster_app.placements

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                self.profiles[cluster][num_nodes[i], num_replicas[i],
                                       local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            if max_num_nodes > 0:
                scalability_selector = (cluster_app.scalability.num_nodes <= max_num_nodes) & (
                    cluster_app.scalability.num_replicas <= max_num_replicas)
                df = cluster_app.scalability[scalability_selector]
            else:
                df = cluster_app.scalability

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
            ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                if profile_key not in profile:
                    profile[profile_key] = step_time[i], sync_time[i]

            # update perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)

    def seed_profiles_rigid(self, cluster_ngpus_per_node):
        print(
            f"Seeding profiles for job: {self.name}, target bsz: {self.target_batch_size}, target num replicas: {self.target_num_replicas}")
        for cluster in cluster_ngpus_per_node.keys():
            cluster_app = self.applications[cluster]
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]
            job_num_nodes = np.ceil(self.target_num_replicas /
                                    cluster_ngpus_per_node[cluster])
            job_num_replicas = self.target_num_replicas

            # add placements data if exists
            placements_selector = (cluster_app.placements.num_nodes == job_num_nodes) & (
                cluster_app.placements.num_replicas <= job_num_replicas)
            df = cluster_app.placements[placements_selector]
            if len(df) == 0:
                print(f"No placements data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num placement profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    self.profiles[cluster][num_nodes[i], num_replicas[i],
                                           local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            scalability_selector = (cluster_app.scalability.num_nodes == job_num_nodes) & (
                cluster_app.scalability.num_replicas <= job_num_replicas)
            df = cluster_app.scalability[scalability_selector]
            if len(df) == 0:
                print(f"No scalability data for job: {self.name}, \
                        cluster: {cluster}, num_nodes: {job_num_nodes}, num_replicas: {job_num_replicas}")
            else:
                print(
                    f"Num scalability profiles: {len(df)} for job: {self.name}")
                num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(
                ), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
                for i in range(len(num_nodes)):
                    profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                    if profile_key not in profile:
                        profile[profile_key] = step_time[i], sync_time[i]
            if not profile:
                print(
                    f"WARNING: No data for job: {self.name}, cluster: {cluster}")
                continue

            # compute perf params for cluster
            self.perf_params[cluster] = self.optimize_perf_params(profile)
            print(
                f"Initialized goodput fn for job: {self.name}, cluster: {cluster}")

    # returns the maximum number of replicas profiled for a given cluster
    def max_profiled_replicas(self, cluster_name=None):
        max_val = 0
        if not cluster_name and self.h_unaware:
            max_val = max((k[1] for k in self.profiles), default=0)
        if cluster_name in self.profiles:
            max_val = max((k[1]
                          for k in self.profiles[cluster_name]), default=0)
        return max_val

    def get_goodput_fn(self, cluster_name=None):
        app = self.applications[cluster_name if cluster_name else "aws"]
        if self.h_unaware:
            perf_params, grad_params = self.perf_params, self.grad_params
        else:
            perf_params, grad_params = self.perf_params[cluster_name], self.grad_params

        # no throughput model yet
        if grad_params is None or perf_params is None:
            return None
        elif app.num_stages > 1:
            return GoodputFunctionPMP(perf_params, grad_params, app.init_batch_size, app.num_stages, app.num_microbatches)
        else:
            return GoodputFunction(perf_params, grad_params, app.init_batch_size)

    def get_speedup_fn(self, cluster_name=None):
        if self.h_unaware:
            if self.grad_params is None:
                return lambda n, r: r
        else:
            perf_params = self.perf_params[cluster_name]
            if self.grad_params is None or perf_params is None:
                return None
        app = self.applications[cluster_name if cluster_name else "aws"]
        max_batch_size = app.max_batch_size if self.target_batch_size is None else self.target_batch_size
        bsz_range = (app.min_local_bsz, app.max_local_bsz)
        return self.speedup_fn_class(self.get_goodput_fn(cluster_name), max_batch_size,
                                     bsz_range, accumulation=True, tune_bsz=self.enable_bsz_tuning)

    # returns throughput of job for different GPU types in examples/sec
    def get_throughputs(self, cluster_ngpus_per_node):
        if self.category != "rigid":
            print(
                f"ERROR:: invalid job category: {self.category} for get_throughput call")
        # cluster throughputs are not computed
        if not self.cluster_throughputs:
            self.cluster_throughputs = dict()
            for cname, cngpus_per_node in cluster_ngpus_per_node.items():
                if self.target_num_replicas <= cngpus_per_node:
                    placement = [self.target_num_replicas]
                else:
                    num_whole_nodes = self.target_num_replicas // cngpus_per_node
                    placement = [cngpus_per_node] * num_whole_nodes
                    need_partial = self.target_num_replicas % cngpus_per_node != 0
                    if need_partial:
                        placement.append(
                            self.target_num_replicas % cngpus_per_node)
                self.cluster_throughputs[cname] = self.applications[cname].get_throughput_with_accum(
                    placement, self.target_batch_size) * self.calibration_factor
            print(
                f"Initialized cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        else:
            print(
                f"Using cached cluster throughputs for job: {self.name} = {self.cluster_throughputs}")
        return {cname: self.cluster_throughputs[cname] for cname in cluster_ngpus_per_node}

    def get_scale_units(self):
        self.scale_units = {
            cluster: app.num_stages for cluster, app in self.applications.items()}
        if sim_config.sia_log_goodputs:
            print(f"Scale units for job: {self.name} = {self.scale_units}")
        return self.scale_units

    # fixes batch size to obey memory/profiling constraints
    # needed for gavel
    def fix_minibatch_size(self):
        print(f"WARNING:: bypassing fix for minibatch size")
        return
        if self.target_num_replicas is not None and self.target_batch_size is not None:
            max_atomic_bsz = math.ceil(
                self.target_batch_size / self.target_num_replicas - 1e-8)
            for cluster, cluster_app in self.applications.items():
                if self.target_num_replicas in cluster_app.placements.num_replicas.values:
                    df = cluster_app.placements[cluster_app.placements.num_replicas ==
                                                self.target_num_replicas]
                    new_bsz = int(min(max_atomic_bsz, df.local_bsz.max()))
                    if new_bsz < max_atomic_bsz:
                        print(
                            f"{self.name}: correcting atomic_bsz: {max_atomic_bsz} -> {new_bsz}")
                        max_atomic_bsz = new_bsz
            target_batch_size = self.target_num_replicas * max_atomic_bsz
            self.target_batch_size = min(
                self.target_batch_size, target_batch_size)

    # call this function to disable bsz tuning
    def disable_bsz_tuning(self):
        self.enable_bsz_tuning = False
        print(f"Disabled bsz tuning for job: {self.name}")

    # given a placement, computes per-GPU batch size, # accumulations
    # uses user-supplied batch size if bsz tuning is disabled
    # NOTE: if bsz tuning disabled ==> the job scales using strong scaling
    def update_local_bsz(self, placement):
        if self.current_cluster is None:
            assert False, "updating local bsz before assigning cluster"
        app = self.applications[self.current_cluster]
        if app.num_stages > 1:
            self.atomic_bsz = 1
            self.accum_steps = app.num_microbatches
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        # use target batch size by default
        batch_size = self.target_batch_size

        # if bsz tuning is enabled, reset batch size and optimize again
        if self.enable_bsz_tuning:
            # print(f"update_local_bsz: bsz tuning ****enabled**** for job: {self.name}")
            batch_size = None
        else:
            # print(f"update_local_bsz: bsz tuning disabled for job: {self.name}")
            pass

        perf_params = self.perf_params if self.h_unaware else self.perf_params[
            self.current_cluster]
        grad_params = self.grad_params
        max_local_bsz = app.get_max_local_bsz(placement)

        # handle PMP jobs gracefully
        if 'pmp' in app.name:
            # obtain pipeline params from app directly
            self.atomic_bsz, self.accum_steps = app.microbatch_size, (
                app.num_microbatches - 1)
            num_stages = app.num_stages
            # each replica runs on num_stages GPUs
            num_replicas = num_replicas // num_stages
            # obtain global bsz using num_replicas
            bsz_per_replica = self.atomic_bsz * (self.accum_steps + 1)
            batch_size = bsz_per_replica * num_replicas
            max_local_bsz = self.atomic_bsz

        # initial batch size if job has not yet started running
        if batch_size is None and (grad_params is None or perf_params is None):
            batch_size = max(app.init_batch_size,
                             app.min_local_bsz * num_replicas)
        # if we have a goodput function, use it to optimize batch size
        if batch_size is None:
            goodput_fn = self.get_goodput_fn(self.current_cluster)
            # use standard goodput function if exists
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, max_local_bsz), accumulation=True)
        else:
            # otherwise, use the batch size specified by the user
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(local_bsz / max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(
                local_bsz / (self.accum_steps + 1) - 1e-8)

        # correct self.atomic_bsz to take into account memory constraints
        if num_replicas in app.placements.num_replicas.values and 'gpt' not in app.name:
            df = app.placements[app.placements.num_replicas == num_replicas]
            new_bsz = int(min(self.atomic_bsz, df.local_bsz.max()))
            if new_bsz < self.atomic_bsz:
                print(
                    f"WARNING #------>{self.name}: correcting atomic_bsz: {self.atomic_bsz} -> {new_bsz}")
                self.atomic_bsz = new_bsz

        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))
        # print(f"update_local_bsz({self.name}): atomic_bsz={self.atomic_bsz}, accum_steps={self.accum_steps},num_replicas={num_replicas}, batch_size={batch_size}")

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        assert self.current_cluster is not None, "current_cluster is None??"
        self.grad_params = (grad_sqr, grad_var)
        if self.h_unaware:
            profile = self.profiles
        else:
            profile = self.profiles[self.current_cluster]

        if (num_nodes, num_replicas, local_bsz) in profile:
            return

        if self.h_unaware:
            self.profiles[num_nodes, num_replicas,
                          local_bsz] = step_time, sync_time
        else:
            self.profiles[self.current_cluster][num_nodes,
                                                num_replicas, local_bsz] = step_time, sync_time

        # get PerfParams for these profiles
        perf_params = self.optimize_perf_params(profile)
        if self.h_unaware:
            self.perf_params = perf_params
        else:
            self.perf_params[self.current_cluster] = perf_params

    def step(self, seconds, interference=0.0):
        if self.completion_time is not None:
            return
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        # job does not use GPUs for `delay` seconds (checkpoint+restore)
        delay = min(self.rescale_time, seconds)
        # step delay time
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.wasted_gpu_seconds += delay * sum(self.placement)
        
        # update rescale time
        self.rescale_time -= delay

        # simulate training for any remaining seconds
        seconds -= delay
        if seconds > 0:
            self.used_gpu_seconds += (seconds * sum(self.placement))
        while seconds > 0 and self.completion_time is None:
            assert self.current_cluster is not None, "stepping on job without current_cluster set"
            application = self.applications[self.current_cluster]
            assert self.epoch < application.max_epochs
            # print(f"job: {self.name}, placement: {self.placement}")
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_gpus = len(placement), sum(placement)
            local_bsz = self.atomic_bsz * (self.accum_steps + 1)
            # one PMP replica can span many stages
            if self.num_stages > 1:
                num_replicas = num_gpus // self.num_stages
            else:
                num_replicas = num_gpus
            # compute minibatch size
            batch_size = num_replicas * local_bsz

            scale = batch_size / application.init_batch_size

            # Calculate true (simulated) efficiency.
            grad_sqr, grad_var = application.get_grad_stats(
                batch_size, self.epoch)
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)

            # Calculate true (simulated) throughput.
            # query xput with atomic_bsz (bsz per pipeline replica)
            query_bsz = self.atomic_bsz
            # check if job is PMP
            if self.num_stages > 1:
                # query xput with local_bsz (bsz per pipeline replica)
                query_bsz = local_bsz
            # get throughput for current placement with query_bsz per replica
            step_time, sync_time = application.get_throughput(
                placement, query_bsz)
            # compute time per accumulation step
            accum_time = step_time - sync_time
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, query_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            # do not count accum steps for PMP, only for DP jobs
            accum_steps = self.accum_steps if self.num_stages == 1 else 0

            # compute time per iter
            total_time = step_time + accum_time * accum_steps
            goodput = gain / total_time * (1.0 - interference)

            # goodput multiplier
            # goodput = self.multiplier * goodput
            # slowdown for job
            goodput = goodput * self.calibration_factor

            # Update current epoch and progress.
            next_progress = application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.execution_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                # update epoch duration
                assert len(self.epoch_duration) == self.epoch
                duration = round(float((application.get_progress(
                    self.epoch + 1) - application.get_progress(self.epoch)) / goodput))
                self.epoch_duration.append(duration)
                if self.epoch == application.max_epochs:
                    print(
                        f"Epoch durations: {self.name} -> {self.epoch_durations}")
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = application.get_completion_epoch(batch_size)
                self.completion_epoch = completion_epoch
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = application.get_best_metric(
                    batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.execution_time += delta
                seconds -= delta
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.

    def reallocate(self, placement):
        old_placement, new_placement = self.placement, tuple(placement)
        if placement:
            if old_placement != new_placement:
                # print(f"RESCALE: job: {self.name}, cluster: {self.current_cluster}, placement: {old_placement} -> {new_placement}")

                # Update placement, num_stages, batch size per replica
                self.num_stages = self.applications[self.current_cluster].num_stages
                self.placement = new_placement
                self.update_local_bsz(self.placement)

                # Start startup/re-scale countdown
                self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
                if self.num_restarts is None:
                    # starting for the first time
                    self.num_restarts = 0
                else:
                    # restarting
                    self.num_restarts += 1
        else:
            # De-allocate all resources.
            # print(f"SUSPEND: job: {self.name}")
            self.placement = ()
            self.atomic_bsz = 0
        
        # Record a rescale event
        new_rescale_event = (self.name, self.current_time, self.current_cluster, old_placement, new_placement, self.rescale_time)
        self.rescale_events.append(new_rescale_event)
        return new_rescale_event

    def migrate(self, new_cluster, new_placement):
        # set current cluster
        prev_cluster = self.current_cluster
        # print(f"MIGRATE:: {self.name}, cluster: {prev_cluster} -> {new_cluster}")
        # update current cluster
        self.current_cluster = new_cluster
        if new_placement:
            self.placement = tuple(new_placement)
            self.update_local_bsz(self.placement)
            # Start startup/re-scale countdown.
            self.rescale_time = self.applications[self.current_cluster].rescale_time or 30
            # print(f"RESCALE_TIME: {self.name} --> {new_cluster}, {self.rescale_time}")
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
            # get num stages for this GPU type
            self.num_stages = self.applications[self.current_cluster].num_stages
        else:
            print(f"SUSPEND: job: {self.name}")
            # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0
        # Record a migrate event
        new_migrate_event = (self.name, self.current_time, prev_cluster, new_cluster, new_placement, self.rescale_time)
        self.migrate_events.append(new_migrate_event)
        return new_migrate_event