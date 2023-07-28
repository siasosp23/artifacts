# python multi_simulator.py ./../no_ncf_workloads/workload1.csv --policy=gavel --interval 360 -gp min_total_duration_perf
import collections
import copy
import math
import cvxpy as cp
import numpy as np
from collections import OrderedDict
import random
from cvxpy.reductions.solvers.defines import SOLVER_MAP_CONIC
import time
import gurobipy

CONFIGS_4GPU = (np.asarray([1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
                np.asarray([1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]))

CONFIGS_8GPU = (np.asarray([1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8]),
                np.asarray([1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64]))

REOPT_ROUNDS = 1


params = {
    "WLSACCESSID": '4befdae3-21b3-4aa3-9868-2913db220ee9',
    "WLSSECRET": '12943e76-6c82-4832-9d55-92a4478f1f39',
    "LICENSEID": 898849,
    "OutputFlag": 0
}
env = gurobipy.Env(params=params)

class ShockWavePolicy(object):
    def __init__(self, future_nrounds, round_duration, solver_rel_gap, solver_num_threads, solver_timeout, logapx_bases, logapx_origin, k, lam, rhomax, aware=False):
        
        # cluster
        self._worker_id_to_cluster_mapping = {}
        self._cluster_to_worker_id_mapping = {}

        self.future_nrounds = future_nrounds
        self.round_duration = round_duration
        self.solver_rel_gap = solver_rel_gap
        self.solver_num_threads = solver_num_threads
        self.solver_timeout = solver_timeout
        self.logapx_bases = logapx_bases
        self.logapx_origin = logapx_origin
        self.k = k
        self.lam = lam
        self.rhomax = rhomax

        self.schedules = OrderedDict()
        self.completed_jobs = OrderedDict()
        self.round_ptr = 1
        self.share_series = {}

        self.resolve = True
        self.reestimate_share = True

        # self.cvxpy_solver = 'GLPK_MI'
        # self.cvxpy_solver = 'ECOS'
        self.cvxpy_solver = getattr(cp, "GUROBI", None)
        assert self.cvxpy_solver is not None and self.cvxpy_solver in cp.installed_solvers()

        # Map from job combinations to assigned workers for current round.
        self._current_worker_assignments = collections.OrderedDict()
        # Map from job combinations to assigned workers for the upcoming round.
        self._next_worker_assignments = None
        self.aware = aware
        assert not aware

    def set_real_cname(self, cluster_num_nodes, cluster_num_gpus):
        self.real_cluster_num_nodes = cluster_num_nodes
        self.real_cluster_num_gpus = cluster_num_gpus

    def populate_valid_configs(self, cluster_num_nodes, cluster_num_gpus):
        assert len(cluster_num_nodes) == 1
        self._cluster_name = list(cluster_num_nodes.keys())
        self._cluster_spec = {}
        for cname in self._cluster_name:
            self._cluster_spec[cname] = cluster_num_nodes[cname] * cluster_num_gpus[cname]
        self._num_gpus_per_server = cluster_num_gpus
        self.register_worker_callback()
        self._cluster_time = {cname: 0 for cname in self._cluster_name}
        self._priorities = {cname: {} for cname in self._cluster_name}
        self._deficits = {cname: {} for cname in self._cluster_name}

        assert len(cluster_num_gpus)==1
        self.cluster_num_nodes = cluster_num_nodes
        self.cluster_num_gpus = cluster_num_gpus
        self.ngpus = np.sum([cluster_num_nodes[cname] * cluster_num_gpus[cname] for cname in cluster_num_gpus])
    
    def register_worker_callback(self):
        i = 0
        for cname in sorted(self._cluster_name):
            for _ in range(self._cluster_spec[cname]):
                self._worker_id_to_cluster_mapping[i] = cname
                i += 1
        j = 0
        n = 0
        for cname in sorted(self._cluster_name):
            self._cluster_to_worker_id_mapping[cname] = []
            num_gpu = self._cluster_spec[cname]
            num_gpu_per_server = self._num_gpus_per_server[cname]
            num_sever = int(num_gpu / num_gpu_per_server)
            for i in range(num_sever):
                self._cluster_to_worker_id_mapping[cname].append(list(range(n + num_gpu_per_server*i, n + num_gpu_per_server*(i+1))))
            j += 1
            n += self._cluster_spec[cname]

        # print("### _worker_id_to_cluster_mapping")
        # print(self._worker_id_to_cluster_mapping)
        # print("### _cluster_to_worker_id_mapping")
        # print(self._cluster_to_worker_id_mapping)

    def convert_worker_ids(self, worker_ids):
        res = []
        cname = self._worker_id_to_cluster_mapping[worker_ids[0]]
        for worker in worker_ids:
            for cid in range(len(self._cluster_name)):
                if cname == self._cluster_name[cid]:
                    idx = worker - sum([self._cluster_spec[cname] for cname in self._cluster_name[:cid]])
                    res.append(math.floor(idx / self._num_gpus_per_server[cname]))
                    # res.append(worker - sum([self._cluster_spec[cname] for cname in self._cluster_name[:cid]]))
        return res

    def finish_time_uniform_share(self, job_infos):
        ngpus = self.ngpus
        njobs = len(job_infos)

        # TODO: check when to reestimate
        # print(f'### compute uniform share')
        # if self.reestimate_share:
        if True:
            for jobid in job_infos:
                job = job_infos[jobid]

                uniform_share = min(1.0, ngpus / njobs)
                assert uniform_share > 0.0

                # fair_share = ngpus / njobs
                # uniform_share = min(1.0, fair_share / job.scale_factor)
                # if uniform_share < 1:
                #     print(f'!!! {jobid} required {job.scale_factor} > fair {fair_share}')
                # assert uniform_share > 0.0



                # job.metadata.calibrate_profiled_epoch_duration()

                # TODO: implement pred_total_runtime
                # finish_time_estimate = (
                #     job.submission_time
                #     + (
                #         sum(job.metadata.epoch_duration[: job.metadata.epoch_progress])
                #         + job.metadata.dirichlet_posterior_remaining_runtime(
                #             job.metadata.epoch_progress
                #         )
                #     )
                #     / uniform_share
                # )
                # finish_time_estimate = (
                    
                #     job.submission_time
                #     + (
                #         job.execution_time
                #         + self.predict_remaining_time(job, jobid)
                #     )
                #     / uniform_share
                # )
                # TODO: LSX check if this is right
                finish_time_estimate = (
                    # self.round_ptr * self.round_duration + self.predict_remaining_time(job, jobid) / uniform_share

                    (self.round_ptr * self.round_duration - job.creation_timestamp + self.predict_remaining_time(job, jobid)) / uniform_share
                    # self.round_duration - job.creation_timestamp % self.round_duration + (self.predict_remaining_time(job, jobid, True)) / uniform_share
                    
                    # job.submission_time
                    # + (
                    #     job.execution_time
                    #     + self.predict_remaining_time(job, jobid)
                    # )
                    # / uniform_share
                )
                # print(f'{jobid}: submission time{job.submission_time} epoch progress: {job.metadata.epoch_progress} sum duration: {job.metadata.epoch_duration[: job.metadata.epoch_progress]} remain: {job.metadata.dirichlet_posterior_remaining_runtime(job.metadata.epoch_progress)}')

                if jobid not in self.share_series:
                    self.share_series[jobid] = []
                self.share_series[jobid].append((self.round_ptr, finish_time_estimate))

        self.reestimate_share = False


    def construct_round_sched_vars(self, njobs):
        jobs_round_sched_vars = []
        for _ in range(njobs):
            jobs_round_sched_vars.append([cp.Variable(boolean=True) for _ in range(self.future_nrounds)])
        return jobs_round_sched_vars

    def construct_round_sched_constraints(self, jobobjs, jobs_round_sched_vars):
        consts = []
        njobs = len(jobs_round_sched_vars)
        assert njobs == len(jobobjs)

        for iround in range(self.future_nrounds):
            round_scheds = []
            for ijob in range(njobs):
                job = jobobjs[ijob]
                nworkers = job.scale_factor
                round_sched_vars = jobs_round_sched_vars[ijob]
                assert len(round_sched_vars) == self.future_nrounds 
                round_scheds.append(cp.multiply(nworkers, round_sched_vars[iround]))
            consts.append(cp.sum(cp.hstack(round_scheds)) <= self.ngpus)

        return consts

    """ iteration based progress"""
    def nash_social_welfare_first_order_apx(self, jobobjs, jobs_round_sched_vars):
        njobs = len(jobobjs)
        assert njobs == len(jobs_round_sched_vars)

        job_log_progresses = []

        assert self.logapx_bases[0] == 0.0
        logapx_base_values = []
        for base in self.logapx_bases:
            assert base >= 0.0 and base <= 1.0
            if base == 0.0:
                assert 0.0 in self.logapx_origin.keys()
                logapx_base_values.append(math.log(self.logapx_origin[0.0]))
            else:
                logapx_base_values.append(math.log(base))
        assert len(self.logapx_bases) == len(logapx_base_values)
        assert all(prev < next for prev, next in zip(logapx_base_values, logapx_base_values[1:]))

        planned_runtime_list = []
        planned_progress_consts = []
        log_apx_consts = []
        planned_progress_vars = []

        for ijob in range(njobs):
            job = jobobjs[ijob]

            """ Iteration based progress"""
            cur_step = job.cur_step
            total_steps = job.total_steps

            planned_progress = cp.Variable(nonneg=True)
            planned_progress_vars.append(planned_progress)
            step_time = self.predict_iter_time_avg_cluster(job, self.jobids[ijob])
            planned_runtime = planned_progress * step_time
            planned_runtime_list.append(planned_runtime)
            planned_progress_consts += [
                planned_runtime <= cp.sum(cp.hstack(jobs_round_sched_vars[ijob])) * self.round_duration
            ]

            objective_progress = cur_step + planned_progress
            objective_progress_normalized = cp.multiply(objective_progress, 1.0 / float(total_steps)) # percentage of completed iters

            vars_cursor = [cp.Variable(nonneg=True) for _ in range(len(self.logapx_bases))]
            var_log_progress_normalized = cp.sum(
                cp.multiply(cp.hstack((vars_cursor)), np.array(logapx_base_values))
            )

            cursor_consts = []
            cursor_consts += [
                cp.sum(
                    cp.multiply(cp.hstack(vars_cursor), np.array(self.logapx_bases))
                )
                == objective_progress_normalized
            ]
            cursor_consts += [cp.sum(cp.hstack(vars_cursor)) == 1.0]

            vars_boundary = [
                cp.Variable(boolean=True) for _ in range(len(self.logapx_bases))
            ]
            boundary_consts = []
            boundary_consts += [cp.sum(cp.hstack(vars_boundary)) <= 2]

            for varcursor, varboundary in zip(vars_cursor, vars_boundary):
                boundary_consts += [varcursor <= varboundary]

            if len(vars_boundary) > 2:
                for lboundary in range(0, len(vars_boundary) - 2):
                    for rboundary in range(lboundary + 2, len(vars_boundary)):
                        boundary_consts += [
                            vars_boundary[lboundary] + vars_boundary[rboundary]<= 1.0
                        ]
            
            log_apx_consts += cursor_consts
            log_apx_consts += boundary_consts
            job_log_progresses.append(var_log_progress_normalized)


        return (
            job_log_progresses,
            planned_runtime_list,
            log_apx_consts,
            planned_progress_consts,
            planned_progress_vars
        )

    def finish_time_momentumed_average(self, series, momentum=0.9):
        assert len(series) > 0
        irounds = [ir for ir, _ in series]
        assert max(irounds) <= self.round_ptr
        irounds += [self.round_ptr]
        ftwindows = np.diff(irounds)
        if max(ftwindows) == 0:
            ftprobs = [1.0]
        else:
            ftprobs = ftwindows / np.sum(ftwindows)
            ftprobs = ftprobs.tolist()
        ftvals = [val for _, val in series]
        assert len(ftprobs) == len(ftvals)
        running_average = 0.0
        for prob, val in zip(ftprobs, ftvals):
            running_average += prob * val

        running_average = (
            momentum * running_average + (1.0 - momentum) * ftvals[-1]
        )

        return running_average

    def call_cvxpy_solver(self, objective, constraints, seed=0):
        result = None
        problem = cp.Problem(objective=objective, constraints=constraints)
        # print("### cvx problems")
        # print(problem)

        random.seed(seed)
        np.random.seed(seed)

        if SOLVER_MAP_CONIC == cp.MOSEK:
            mosek_options = {
                cp.dparam.mio_tol_rel_gap: self.solver_rel_gap,
                cp.iparam.num_threads: self.solver_num_threads,
                cp.dparam.optimizer_max_time: self.solver_timeout,
            }
            result = problem.solve(
                solver=cp.MOSEK, verbose=True, mosek_params=mosek_options
            )
        elif self.cvxpy_solver == cp.GUROBI:
            time_limit = (
                self.solver_timeout if self.solver_timeout > 0 else gurobipy.GRB.INFINITY
            )
            result = problem.solve(
                solver=cp.GUROBI,
                env = env,
                verbose=False,
                MIPGap=self.solver_rel_gap,
                Threads=self.solver_num_threads,
                TimeLimit=time_limit,
            )
        else:
            print("CVXPY solver is not supported.")
            exit()

        return problem, result



        result = problem.solve(solver=cp.GLPK_MI, glpk={'msg_lev': 'GLP_MSG_OFF'}, MIPGap=self.solver_rel_gap, Threads=self.solver_num_threads, verbose=False)
        # result = problem.solve(solver=self.cvxpy_solver, MIPGap=self.solver_rel_gap, Threads=self.solver_num_threads, verbose=False)

        return problem, result

    def relax_finish_time_constraints(self, jobobjs, jobutils, share_series):
        
        # print('*'*40, 'utility')
        # for ijob in range(len(jobobjs)):
        #     print(self.jobids[ijob], jobutils[ijob])
        # print('*'*40)

        priority_threshold = self.rhomax
        priority_power = self.lam
        priority_M = 1e2

        njobs = len(jobobjs)
        assert self.ngpus > 0
        assert njobs > 0

        jobs_rho_ratios = []
        jobs_remaining_runtime_projected = []

        print(f'\n### finish time fairness')
        for ijob in range(len(jobobjs)):
            job = jobobjs[ijob]
            shares = share_series[ijob]
            round_time = self.round_duration * self.round_ptr

            future_share = min(1.0, self.ngpus / njobs)
            # fair_share = self.ngpus / njobs
            # future_share = min(1.0, fair_share / job.scale_factor)
            assert future_share > 0.0

            # job.metadata.calibrate_profiled_epoch_duration()
            # remaining_runtime_projected = job.metadata.dirichlet_posterior_remaining_runtime()
            remaining_runtime_projected = self.predict_remaining_time(job, self.jobids[ijob])

            jobs_remaining_runtime_projected.append(remaining_runtime_projected)
            # TODO: LSX check if this is correct
            # finish_time_projected = (
            #     round_time + remaining_runtime_projected / future_share
            # )
            assert round_time - job.creation_timestamp >= 0
            finish_time_projected = (
                round_time - job.creation_timestamp + remaining_runtime_projected / future_share
            )
            

            finish_time_runavg = self.finish_time_momentumed_average(shares)

            rho_ratio_projected = finish_time_projected / finish_time_runavg

            print(f'{self.jobids[ijob]}: {finish_time_projected} / {finish_time_runavg} = {rho_ratio_projected}')


            jobs_rho_ratios.append(rho_ratio_projected)

        jobs_log_utilities_upgrade = []
        jobs_priorities = []

        for ijob, ratio in enumerate(jobs_rho_ratios):
            log_utility = jobutils[ijob]
            job = jobobjs[ijob]

            remaining_runtime_projected = jobs_remaining_runtime_projected[ijob]
            if ratio > priority_threshold:
                # if self.jobids[ijob].split('-')[0] == 'imagenet':
                #     pass
                #     priority = 1.0
                # elif self.jobids[ijob].split('-')[0] == 'cifar10' or self.jobids[ijob].split('-')[0] == 'ncf':
                #     pass
                #     priority = ratio ** 100
                # else:
                #     priority = ratio ** priority_power
                #     if remaining_runtime_projected < self.round_duration:
                #         priority = ratio ** priority_M

                priority = ratio ** priority_power
                if remaining_runtime_projected < self.round_duration:
                    print(f"!!!!!!!!!!!!!!! {self.jobids[ijob]} can finish in this round", remaining_runtime_projected, self.round_duration)
                    priority = ratio ** priority_M
                    # priority = 1e10

                utility_upgrade = log_utility * priority
                print(f"Relax finish time constraints for Job {self.jobids[ijob]}, {ratio} ** {math.log(priority, ratio)} = Priority: {priority}")
            else:
                priority = 1.0
                utility_upgrade = log_utility
            jobs_log_utilities_upgrade.append(utility_upgrade)
            jobs_priorities.append(priority)

        return jobs_log_utilities_upgrade, jobs_priorities

    def rank_in_schedule_jobs(self, jobs_round_sched_vars, jobs_priorities, jobobjs):
        njobs = len(jobs_round_sched_vars)
        nrounds = len(jobs_round_sched_vars[0])
        jobs_nworkers = [job.scale_factor for job in jobobjs]

        jobs_round_sched_ranked = []
        for _ in range(njobs):
            jobs_round_sched_ranked.append([cp.Variable(boolean=True) for _ in range(nrounds)])

        jobs_sched_nrounds = []
        for ijob in range(njobs):
            jobs_sched_nrounds.append(cp.sum(cp.hstack(jobs_round_sched_vars[ijob])).value)

        consts = []
        for ijob in range(njobs):
            consts += [jobs_sched_nrounds[ijob] == cp.sum(cp.hstack(jobs_round_sched_ranked[ijob]))]

        for iround in range(nrounds):
            scheds = [jobs_round_sched_ranked[ijob][iround] for ijob in range(njobs)]
            consts += [cp.sum(cp.multiply(cp.hstack(scheds), cp.hstack(jobs_nworkers)))<= self.ngpus]

        obj_components = []
        for ijob in range(njobs):
            priority = jobs_priorities[ijob]
            if jobs_sched_nrounds[ijob] > 0:
                avg_shed_idx = (
                    cp.sum(
                        cp.multiply(
                            cp.hstack([t for t in range(nrounds)]),
                            cp.hstack(jobs_round_sched_ranked[ijob]),
                        )
                    )
                    / jobs_sched_nrounds[ijob]
                )
                obj_components.append(avg_shed_idx * priority)

        if len(obj_components) <= 0:
            return jobs_round_sched_vars

        obj = cp.Minimize(cp.sum(cp.hstack(obj_components)))

        problem, result = self.call_cvxpy_solver(obj, consts)

        if problem.status not in cp.settings.SOLUTION_PRESENT:
            return jobs_round_sched_vars

        return jobs_round_sched_ranked


    def dynamic_eisenberg_gale_scheduling(self, jobids, jobobjs, share_series):

        njobs = len(jobobjs)
        constraints = []

        jobs_round_sched_vars = self.construct_round_sched_vars(njobs)
        constraints += self.construct_round_sched_constraints(jobobjs, jobs_round_sched_vars)

        # print("*"*10, "var and constraints")
        # print(len(jobs_round_sched_vars), len(jobs_round_sched_vars[0]))
        # for cons in constraints:
        #     print(cons)
        
        (
            jobs_log_utilities,
            jobs_planned_runtime,
            swconsts,
            progress_consts,
            planned_progress_vars
        ) = self.nash_social_welfare_first_order_apx(jobobjs, jobs_round_sched_vars)


        jobs_remaining_time_sched = []
        for ijob in range(len(jobs_planned_runtime)):
            job = jobobjs[ijob]
            # remaining_runtime_shed = cp.maximum(
            #     0,
            #     job.metadata.dirichlet_posterior_remaining_runtime() # P(j)
            #     - jobs_planned_runtime[ijob],
            # )
            remaining_runtime_shed = cp.maximum(
                0,
                self.predict_remaining_time(job, jobids[ijob])
                - jobs_planned_runtime[ijob],
            )
            jobs_remaining_time_sched.append(remaining_runtime_shed)


        objective = cp.Maximize(
            cp.sum(cp.hstack(jobs_log_utilities) / (njobs * self.future_nrounds)) - self.k * cp.max(cp.hstack(jobs_remaining_time_sched))
        )
        constraints += swconsts
        constraints += progress_consts

        finish_time_consts = []
        next_sched_time = self.round_duration * (self.round_ptr + self.future_nrounds)

        finish_time_sched_vars = []
        finish_time_objectives = []

        for ijob in range(len(share_series)):
            job = jobobjs[ijob]

            future_share = min(1.0, self.ngpus / njobs)
            # fair_share = self.ngpus / njobs
            # future_share = min(1.0, fair_share / job.scale_factor)
            assert future_share > 0.0
            
            round_time = (self.round_duration) * self.round_ptr

            remaining_runtime_shed = jobs_remaining_time_sched[ijob]
            # finish_time_sched = next_sched_time + cp.multiply(remaining_runtime_shed, 1.0 / future_share)
            # TODO : LSX check if this is correct
            assert round_time - job.creation_timestamp >= 0, f'{round_time} {job.creation_timestamp}'
            finish_time_sched = next_sched_time - job.creation_timestamp + cp.multiply(remaining_runtime_shed, 1.0 / future_share)
            # finish_time_sched = self.round_ptr * self.round_duration - job.creation_timestamp + jobs_planned_runtime[ijob] + cp.multiply(remaining_runtime_shed, 1.0 / future_share)
            finish_time_objective = self.finish_time_momentumed_average(share_series[ijob])

            

            finish_time_consts.append(finish_time_sched <= finish_time_objective * self.rhomax)

            finish_time_sched_vars.append(finish_time_sched)
            finish_time_objectives.append(finish_time_objective)

        ENABLE_FTF_CONSTS = True

        if ENABLE_FTF_CONSTS:
            st_time = time.time()
            problem, result = self.call_cvxpy_solver(
                objective,
                constraints + finish_time_consts,
            )
            ed_time = time.time()

        if ENABLE_FTF_CONSTS and problem.status in cp.settings.SOLUTION_PRESENT:
            # print(f'\n### find optimal solution')
            # print(f'\n### planned progress')
            # for ijob in range(len(jobobjs)):
            #     progres = planned_progress_vars[ijob].value
            #     iter_time = self.predict_iter_time_avg_cluster(jobobjs[ijob])

            #     print(f'{self.jobids[ijob]} planned_progres: {progres} * {iter_time} = {progres * iter_time} {jobs_planned_runtime[ijob].value}')

            # print("*"*10, f"Final optimal sol with finish time constraints...")
            self.solver_solution_info(jobids, jobobjs,jobs_round_sched_vars, finish_time_sched_vars, finish_time_objectives, jobs_remaining_time_sched, jobs_log_utilities, jobs_planned_runtime, share_series)
            
            # print(f'\n### metric')
            # ftf_metric = cp.sum(
            #         cp.hstack(jobs_log_utilities)
            #         / (njobs * self.future_nrounds)
            #     ).value
            # makespan_metric = - self.k * cp.max(cp.hstack(jobs_remaining_time_sched)).value
            # print(f'ftf metric: {ftf_metric} makespan metric: {makespan_metric}')


        else:
            if not ENABLE_FTF_CONSTS or problem.status in cp.settings.INF_OR_UNB:
                # print(f"\n### Nullify finish time constraints...")
                (
                    jobs_log_utilities_upgrade,
                    jobs_priorities,
                ) = self.relax_finish_time_constraints(
                    jobobjs,
                    jobs_log_utilities,
                    share_series,
                )

                objective = cp.Maximize(
                    cp.sum(
                        cp.hstack(jobs_log_utilities_upgrade)
                        / (njobs * self.future_nrounds)
                    )
                    - self.k * cp.max(cp.hstack(jobs_remaining_time_sched))
                )

                problem, result = self.call_cvxpy_solver(
                    objective,
                    constraints
                )

                assert problem.status in cp.settings.SOLUTION_PRESENT                
                # print('\nbefore rank')

                # print("*"*10, f"Optimize job ranks in schedule...")
                jobs_round_sched_vars = self.rank_in_schedule_jobs(jobs_round_sched_vars, jobs_priorities, jobobjs)  
                # print("*"*10, f"Adjusted job ranks in schedule:")
                self.solver_solution_info(jobids, jobobjs,jobs_round_sched_vars, finish_time_sched_vars, finish_time_objectives, jobs_remaining_time_sched, jobs_log_utilities_upgrade, jobs_planned_runtime, share_series)

                # print(f'\n### remaing time after this window')
                # for ijob in range(len(jobobjs)):
                #     print(f'{self.jobids[ijob]}: {jobs_remaining_time_sched[ijob].value}')
                    
                # print(f'\n### planned progress')
                # for ijob in range(len(jobobjs)):
                #     progres = planned_progress_vars[ijob].value
                #     iter_time = self.predict_iter_time_avg_cluster(jobobjs[ijob])
                #     print(f'{self.jobids[ijob]} planned_progres: {progres} * {iter_time} = {progres * iter_time} {jobs_planned_runtime[ijob].value}')

                
                # print(f'\n### metric')
                # ftf_metric = cp.sum(
                #         cp.hstack(jobs_log_utilities_upgrade)
                #         / (njobs * self.future_nrounds)
                #     ).value
                # makespan_metric = - self.k * cp.max(cp.hstack(jobs_remaining_time_sched)).value
                # print(f'ftf metric: {ftf_metric} makespan metric: {makespan_metric}')


            else:
                print(f"Solver internal error.")
                exit()

        return jobs_round_sched_vars
    
    # only the jobids who are shceduled
    def construct_schedules(self, schedule_solution, jobids, jobobjs):
        njobs = len(jobids)
        nrounds = len(schedule_solution[0])
        assert njobs == len(jobobjs)
        assert njobs == len(schedule_solution)
        assert nrounds == self.future_nrounds

        rounds_sched = OrderedDict()

        for iround in range(nrounds):
            round_index = self.round_ptr + iround
            jobids_cur_round = []
            jobobjs_cur_round = []
            for ijob in range(njobs):
                job = jobobjs[ijob]
                jobid = jobids[ijob]
                sched = schedule_solution[ijob][iround].value[()]
                if round(sched) == 1.0:
                    jobids_cur_round.append(jobid)
                    jobobjs_cur_round.append(job)
            if len(jobids_cur_round) <= 0:
                print("!!!!!!!!!!! Invalid solution: none jobs are scheduled in round {}".format(round_index))

            nworkers = sum([job.scale_factor for job in jobobjs_cur_round])
            n_idle_workers = self.ngpus - nworkers

            if n_idle_workers > 0:
                non_sched_indice = [
                    idx
                    for idx in range(njobs)
                    if jobids[idx] not in jobids_cur_round
                ]
                non_sched_indice_sorted = sorted(
                    non_sched_indice,
                    # key=lambda idx:  jobobjs[idx].metadata.dirichlet_posterior_remaining_runtime(),
                    key=lambda idx:  self.predict_remaining_time(jobobjs[idx], jobids[idx]),
                    reverse=True,
                )

                for ijob in non_sched_indice_sorted:
                    jobid = jobids[ijob]
                    job = jobobjs[ijob]
                    if job.scale_factor <= n_idle_workers:
                        n_idle_workers -= job.scale_factor
                        jobids_cur_round.append(jobid)
                        # print("Work conserving scheduling for job {}".format(jobid))
                    if n_idle_workers <= 0:
                        break
            rounds_sched[round_index] = jobids_cur_round

        return rounds_sched

    def assign_workers_to_job(self, job_id, scale_factor, worker_state, worker_assignments):
        worker_ids = worker_state['worker_ids']
        assigned_worker_ids = worker_state['assigned_worker_ids']
        server_id_ptr = worker_state['server_id_ptr']

        if job_id in worker_assignments:
            worker_ids_for_job = list(worker_assignments[job_id])
        else:
            worker_ids_for_job = []
        while len(worker_ids_for_job) < scale_factor and server_id_ptr < len(worker_ids):
            if len(worker_ids[server_id_ptr]) == 0:
                server_id_ptr += 1
                continue
            worker_id_to_assign = worker_ids[server_id_ptr][0]
            if worker_id_to_assign not in assigned_worker_ids:
                worker_ids_for_job.append(worker_id_to_assign)
                assigned_worker_ids.add(worker_id_to_assign)
            worker_ids[server_id_ptr].pop(0)
        
        if len(worker_ids_for_job) != scale_factor:
            raise RuntimeError('Could not assign workers to job %s!' % (job_id))

        worker_assignments[job_id] = tuple(worker_ids_for_job)
        worker_state['server_id_ptr'] = server_id_ptr

    def schedule_jobs_on_workers_helper(self, job_infos):
        cname = list(self.cluster_num_gpus.keys())[0]
        scheduled_jobs = {cname: []}

        job_ids = self.schedules[self.round_ptr]
        for job_id in job_ids:
            scale_factor = job_infos[job_id].scale_factor
            scheduled_jobs[cname].append((job_id, scale_factor))

        return scheduled_jobs

    def schedule_jobs_on_workers(self, job_infos):
        new_worker_assignments = collections.OrderedDict()

        scheduled_jobs = self.schedule_jobs_on_workers_helper(job_infos)

        # print("### selected jobs:", sum([len(v) for _, v in scheduled_jobs.items()]), "/", len(job_infos))
        # for cname, v in scheduled_jobs.items():
        #     print(cname, v)

        cluster_state = {}
        for cname in self._cluster_name:
            scheduled_jobs[cname].sort(key=lambda x: x[1], reverse=True)
            worker_ids = copy.deepcopy(self._cluster_to_worker_id_mapping[cname])
            cluster_state[cname] = {
                'worker_ids': worker_ids,
                'assigned_worker_ids': set(),
                'server_id_ptr': 0,
            }

        prev_cluster_types = {}
        for (job_id, worker_ids) in self._current_worker_assignments.items():
            cname = self._worker_id_to_cluster_mapping[worker_ids[0]]
            prev_cluster_types[job_id] = cname

        for cname in self._cluster_name:
            per_cluster_state = cluster_state[cname]
            assigned_worker_ids = per_cluster_state['assigned_worker_ids']


            scale_factors = set(x[1] for x in scheduled_jobs[cname])
            scale_factors = sorted(scale_factors, reverse=True)

            for current_scale_factor in scale_factors:
                # Try to keep jobs on current workers if possible.
                for (job_id, scale_factor) in scheduled_jobs[cname]:
                    if scale_factor != current_scale_factor:
                        continue
                    if job_id in prev_cluster_types and prev_cluster_types[job_id] == cname:
                        prev_worker_ids = self._current_worker_assignments[job_id]
                        assert(isinstance(prev_worker_ids, tuple))
                        extend_placement = True
                        for prev_worker_id in prev_worker_ids:
                            if prev_worker_id in assigned_worker_ids:
                                extend_placement = False
                                break
                        if extend_placement:
                            new_worker_assignments[job_id] = prev_worker_ids
                            for prev_worker_id in prev_worker_ids:
                                assigned_worker_ids.add(prev_worker_id)

                # Assign workers for remaining jobs.
                for job_id, scale_factor in scheduled_jobs[cname]:
                    if scale_factor != current_scale_factor:
                        continue

                    self.assign_workers_to_job(job_id, scale_factor,
                                                per_cluster_state,
                                                new_worker_assignments)
        
        # Verify the assignment.
        num_assignments = {}
        for job_id in new_worker_assignments:
            for worker_id in new_worker_assignments[job_id]:
                if worker_id not in num_assignments:
                    num_assignments[worker_id] = 0
                num_assignments[worker_id] += 1
        for worker_id in num_assignments:
            if num_assignments[worker_id] != 1:
                raise RuntimeError('Worker {0} was assigned {1} times!'.format(worker_id, num_assignments[worker_id]))

        return new_worker_assignments

    def optimize(self, job_infos, nodes, prev_allocations, node_info):
        print(f"########################## Schedule for next round: {self.round_ptr} ##################################")
        # print(f'### resolve: {self.resolve} scheduled: {self.round_ptr in self.schedules.keys()}')


        jobids = list(job_infos.keys())
        jobobjs = list(job_infos.values())
        self.jobids = jobids
        self.jobobjs = jobobjs

        # # use previous allocations
        if not self.resolve and len(self.schedules) > 0 and self.round_ptr in self.schedules:
            # pass
            print(f'### use previous solns')
            # solution =  self.schedules[self.round_ptr]
        else:
            # print('### throughput_timeline')
            # for jid, job in job_infos.items():
            #     print(f'{jid}: {job.metadata.throughput_measurements}')

            self.finish_time_uniform_share(job_infos)
            share_series = [self.share_series[jobid] for jobid in jobids]

            # print('\n### share seris')
            # for ijob in range(len(jobids)):
            #     print(f'{jobids[ijob]}: {share_series[ijob]}')

            solution = self.dynamic_eisenberg_gale_scheduling(jobids, jobobjs, share_series)

            # print('### ilp solution')
            # for ijob in range(len(solution)):
            #     print(f'{jobids[ijob]}: {[v.value for v in solution[ijob]]}')

            self.schedules = self.construct_schedules(solution, jobids, jobobjs)

        self._current_worker_assignments = self.schedule_jobs_on_workers(job_infos)

        # print("### new worker assignments:")
        # print(self._current_worker_assignments)

        res = {}
        for job_id, worker_ids in self._current_worker_assignments.items():
            cname = self._worker_id_to_cluster_mapping[worker_ids[0]]
            ids = self.convert_worker_ids(worker_ids)
            if self.aware:
                res[job_id] = (cname, ids) 
            else:
                res[job_id] = ids

        # print("\n### converted_ids:")  
        # print(res)     

        self.resolve = False

        if (self.round_ptr + 1) % REOPT_ROUNDS == 0:
            # print(f'### rescheduler every {REOPT_ROUNDS} rounds')
            self.resolve = True

        self.round_ptr += 1
        
        print("########################## End ##################################")   
        return res, None

    


    def solver_solution_info(
        self,
        jobids,
        jobobjs,
        jobs_round_sched_vars,
        finish_time_sched_vars,
        finish_time_objectives,
        jobs_remaining_time_sched,
        job_log_utilities,
        jobs_planned_run_time,
        share_series,
    ):
        return 

        # print('### ilp solution')
        # for ijob in range(len(jobs_round_sched_vars)):
        #     print(f'{jobids[ijob]}: {[v.value for v in jobs_round_sched_vars[ijob]]}')

        for ijob in range(len(jobobjs)):
            job = jobobjs[ijob]
            shares = share_series[ijob]
            round_sched = [var.value for var in jobs_round_sched_vars[ijob]]
            sched_ftf = float(finish_time_sched_vars[ijob].value)
            runavg_ftf = float(finish_time_objectives[ijob])

            # print(f"#" * 20)
            print(f"-" * 80, 'solution info')
            print(f"Job {jobids[ijob]} nworker: {job.scale_factor}")
            print(f"Current Round:{self.round_ptr}")

            avg_iter_time = self.predict_iter_time_avg_cluster(job, jobids[ijob])
            remaining_step = job.total_steps - job.cur_step
            res = remaining_step * avg_iter_time

            print(f'Remaining step: {job.total_steps} - {job.cur_step} = {remaining_step}  remaining time: {remaining_step} * {avg_iter_time} = {remaining_step*avg_iter_time} finish time: {self.round_ptr * self.round_duration} - {job.creation_timestamp} + {res} = {self.round_ptr * self.round_duration - job.creation_timestamp + res}')

            print(f"Round schedule:{round_sched}")
            print(f'Planned Runtime: {jobs_planned_run_time[ijob].value}')

            print(f"Finish Time (VAR.VALUE): {self.round_duration*(self.round_ptr+self.future_nrounds)} - {job.creation_timestamp} + {jobs_remaining_time_sched[ijob].value} = {sched_ftf}")
            print(f"Finish Time Objective (RunAvg): {runavg_ftf}")
            print(f"Ratio: {sched_ftf/runavg_ftf}")
            print(f"-" * 80)
        
        ftf_metric = cp.sum(
                cp.hstack(job_log_utilities)
                / (len(jobobjs) * self.future_nrounds)
            ).value
        makespan_metric = - self.k * cp.max(cp.hstack(jobs_remaining_time_sched)).value
        print(f'Solution progress metric: {ftf_metric} makespan metric: {makespan_metric}')
        print(f"-" * 80)
            



    def predict_iter_time(self, job, cname, jobname):
        placement = ()
        num_gpu_per_node = self.real_cluster_num_gpus[cname]
        # no enough gpus in this cluster
        if job.scale_factor > self.real_cluster_num_gpus[cname] * self.real_cluster_num_nodes[cname]:
            assert False
            return 1e-1
        while sum(placement) < job.scale_factor:
            placement = (*placement, min(job.scale_factor - sum(placement), num_gpu_per_node))

        local_bsz = math.ceil(job.target_batch_size / job.scale_factor - 1e-8)
        accum_steps = math.ceil(
            local_bsz / job.applications[cname].max_local_bsz - 1e-8) - 1
        if job.scale_factor == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = job.scale_factor * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(
            job.applications[cname].max_batch_size / count))
        #throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        # return atomic_bsz * count / throughput
        step_time, sync_time = job.applications[cname].get_throughput(
            placement, atomic_bsz)
        assert step_time > 0 and sync_time > 0, f'{jobname} accum_steps: {local_bsz} / {job.applications[cname].max_local_bsz} = {accum_steps} atomic_bsz: {atomic_bsz} placement: {placement} cname: {cname}'
        # apply slow down factor
        assert job.slowdown_factor < 1
        return (step_time + (step_time - sync_time) * accum_steps) / job.slowdown_factor


        # local_bsz = math.ceil(job.target_batch_size / job.scale_factor - 1e-8)
        # accum_steps = math.ceil(local_bsz / job.applications[cname].max_local_bsz - 1e-8) - 1
        # if job.scale_factor == 1:
        #     accum_steps = max(1, accum_steps)
        # atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        # count = job.scale_factor * (accum_steps + 1)
        # atomic_bsz = min(atomic_bsz, int(job.applications[cname].max_batch_size / count))
        # step_time, sync_time = job.applications[cname].get_throughput(placement, atomic_bsz)
        # assert step_time > 0 and sync_time > 0
        # return (step_time + (step_time - sync_time) * accum_steps)

    def predict_iter_time_avg_cluster(self, job, jobname):
        avg_iter_time = []
        for cname in self.real_cluster_num_nodes:
            iter_time = self.predict_iter_time(job, cname, jobname)
            avg_iter_time.append(iter_time)
        return np.mean(avg_iter_time)

    def predict_remaining_time(self, job, jobname, total_time=False):
        # TODO: now use average throughput across all clusters

        avg_iter_time = self.predict_iter_time_avg_cluster(job, jobname)

        remaining_step = job.total_steps - job.cur_step if not total_time else job.total_steps
        res = remaining_step * avg_iter_time

        # if jobname is not None:
        #     print("*"*80, 'predict remained time')
        #     print(f'{jobname} remaining step: {job.total_steps} - {job.cur_step} = {remaining_step}  remaining time: {remaining_step} * {avg_iter_time} = {remaining_step*avg_iter_time} finish time: {self.round_ptr * self.round_duration - job.creation_timestamp + res}')
        #     print("*"*80)
        return res