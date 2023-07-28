import csv
from datetime import datetime
import json
import os
import pickle
import random
import re
import socket
import subprocess
'''
from adaptdl_sched.policy.gavel_policies import allox, fifo, finish_time_fairness, gandiva, isolated, \
    max_min_fairness, max_min_fairness_water_filling, max_sum_throughput, \
    min_total_duration, min_jct
'''
from adaptdl_sched.policy.gavel_policies import max_sum_throughput

# from job import Job

def get_available_policies():
    return ['allox',
            'fifo', 'fifo_perf', 'fifo_packed',
            'finish_time_fairness',
            'finish_time_fairness_perf',
            'finish_time_fairness_packed',
            'gandiva',
            'isolated',
            'max_min_fairness',
            'max_min_fairness_perf',
            'max_min_fairness_packed',
            'max_min_fairness_water_filling',
            'max_min_fairness_water_filling_perf',
            'max_min_fairness_water_filling_packed',
            'max_sum_throughput_perf',
            'max_sum_throughput_normalized_by_cost_perf',
            'max_sum_throughput_normalized_by_cost_perf_SLOs',
            'max_sum_throughput_normalized_by_cost_packed_SLOs',
            'min_total_duration',
            'min_total_duration_perf',
            'min_total_duration_packed',
            'min_jct_perf'
            ]

def parse_job_type_tuple(job_type):
    match = re.match('\(\'(.*)\', (\d+)\)', job_type)
    if match is None:
        return None
    model = match.group(1)
    scale_factor = int(match.group(2))
    return (model, scale_factor)

# def stringify_throughputs(throughputs):
#     stringified_throughputs = {}
#     for worker_type in throughputs:
#         stringified_throughputs[worker_type] = {}
#         for key in throughputs[worker_type]:
#             stringified_throughputs[worker_type][str(key)] = {}
#             for other_key in throughputs[worker_type][key]:
#                 stringified_throughputs[worker_type][str(key)][str(other_key)] = \
#                     throughputs[worker_type][key][other_key]
#     return stringified_throughputs

def read_all_throughputs_json_v2(file_name):
    with open(file_name, 'r') as f:
        raw_throughputs = json.load(f)
    parsed_throughputs = {}
    for worker_type in raw_throughputs:
        parsed_throughputs[worker_type] = {}
        for job_type in raw_throughputs[worker_type]:
            key = parse_job_type_tuple(job_type)
            assert(key is not None)
            parsed_throughputs[worker_type][key] = {}
            for other_job_type in raw_throughputs[worker_type][job_type]:
                if other_job_type == 'null':
                    other_key = other_job_type
                else:
                    other_key = parse_job_type_tuple(other_job_type)
                    assert(other_key is not None)
                
                parsed_throughputs[worker_type][key][other_key] =\
                    raw_throughputs[worker_type][job_type][other_job_type]
                # print(worker_type, key, other_key, parsed_throughputs[worker_type][key][other_key])
    return parsed_throughputs

# def read_all_throughputs_json(throughputs_file):
#     with open(throughputs_file, 'r') as f:
#         throughputs = json.load(f)
#     return throughputs

def get_policy(policy_name, solver=None, seed=None,
               priority_reweighting_policies=None):
    if policy_name.startswith('allox'):
        if policy_name == 'allox':
            alpha = 1.0
        else:
            alpha = float(policy_name.split("allox_alpha=")[1])
        policy = allox.AlloXPolicy(alpha=alpha)
    elif policy_name == 'fifo':
        policy = fifo.FIFOPolicy(seed=seed)
    elif policy_name == 'fifo_perf':
        policy = fifo.FIFOPolicyWithPerf()
    elif policy_name == 'fifo_packed':
        policy = fifo.FIFOPolicyWithPacking()
    elif policy_name == 'finish_time_fairness':
        policy = finish_time_fairness.FinishTimeFairnessPolicy(solver=solver)
    elif policy_name == 'finish_time_fairness_perf':
        policy = \
            finish_time_fairness.FinishTimeFairnessPolicyWithPerf(solver=solver)
    elif policy_name == 'finish_time_fairness_packed':
        policy = \
            finish_time_fairness.FinishTimeFairnessPolicyWithPacking(
                solver=solver)
    elif policy_name == 'gandiva':
        policy = gandiva.GandivaPolicy(seed=seed)
    elif policy_name == 'isolated':
        policy = isolated.IsolatedPolicy()
    elif policy_name == 'max_min_fairness':
        policy = max_min_fairness.MaxMinFairnessPolicy(solver=solver)
    elif policy_name == 'max_min_fairness_perf':
        policy = max_min_fairness.MaxMinFairnessPolicyWithPerf(solver=solver)
    elif policy_name == 'max_min_fairness_packed':
        policy = \
            max_min_fairness.MaxMinFairnessPolicyWithPacking(solver=solver)
    elif policy_name == 'max_min_fairness_water_filling':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicy(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_min_fairness_water_filling_perf':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPerf(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_min_fairness_water_filling_packed':
        policy = max_min_fairness_water_filling.MaxMinFairnessWaterFillingPolicyWithPacking(
            priority_reweighting_policies=priority_reweighting_policies)
    elif policy_name == 'max_sum_throughput_perf':
        policy = max_sum_throughput.ThroughputSumWithPerf(solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_perf':
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerf(
                    solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_perf_SLOs':
        policy = max_sum_throughput.ThroughputNormalizedByCostSumWithPerfSLOs(
                    solver=solver)
    elif policy_name == 'max_sum_throughput_normalized_by_cost_packed_SLOs':
        policy = \
            max_sum_throughput.ThroughputNormalizedByCostSumWithPackingSLOs(
                                                        solver=solver)
    elif policy_name == 'min_total_duration':
        policy = min_total_duration.MinTotalDurationPolicy(solver=solver)
    elif policy_name == 'min_total_duration_perf':
        policy = min_total_duration.MinTotalDurationPolicyWithPerf(solver=solver)
    elif policy_name == 'min_total_duration_packed':
        policy = \
            min_total_duration.MinTotalDurationPolicyWithPacking(solver=solver)
    elif policy_name == 'min_jct_perf':
        policy = min_jct.MinJCTPerf(solver=solver)
    else:
        raise ValueError('Unknown policy!')
    return policy
'''
def parse_trace(trace_file):
    jobs = []
    arrival_times = []
    with open(trace_file, 'r') as f:
        for line in f:
            (job_type, command, working_directory, num_steps_arg,
             needs_data_dir, total_steps, scale_factor, priority_weight, SLO,
             arrival_time) = line.split('\t')
            assert(int(scale_factor) >= 1)
            jobs.append(Job(job_id=None,
                            job_type=job_type,
                            command=command,
                            working_directory=working_directory,
                            needs_data_dir=bool(int(needs_data_dir)),
                            num_steps_arg=num_steps_arg,
                            total_steps=int(total_steps),
                            duration=None,
                            scale_factor=int(scale_factor),
                            priority_weight=float(priority_weight),
                            SLO=float(SLO)))
            arrival_times.append(float(arrival_time))
    return jobs, arrival_times
'''

def print_allocation(allocation, current_time=None):
    """Prints the allocation.

       Debug method used for printing the allocation of each job on each
       worker type.
    """
    print('=' * 80)
    if current_time is not None:
        print('Allocation\t(Current_time: %f)' % (current_time))
        print('-' * 80)
    for job_id in sorted(list(allocation.keys())):
        allocation_str = 'Job ID %s:' % (job_id)
        for worker_type in sorted(list(allocation[job_id].keys())):
            value = allocation[job_id][worker_type]
            allocation_str += ' [%s: %f]' % (worker_type, value)
        print(allocation_str)
    print('=' * 80)
