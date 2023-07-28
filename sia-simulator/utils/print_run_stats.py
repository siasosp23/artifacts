import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import glob
import argparse

SCHED_INTERVAL = 60


def read_logs(logfile):
    print(f"Reading {logfile}")
    logs = []
    with open(logfile, "r") as f:
        for line in f:
            entry = json.loads(line)
            log_entry = {}
            log_entry["t"] = entry["timestamp"]
            running, queued = 0, 0
            gpu_seconds = {}
            for job_entry in entry["submitted_jobs"]:
                if job_entry["completion_time"] is None:
                    job_gpu_seconds = {}
                    if job_entry["allocation"] != []:
                        gpu_type, gpu_list = job_entry["allocation"]
                        if gpu_type is None:
                            queued += 1
                            continue
                        running += 1
                        if gpu_type not in gpu_seconds:
                            job_gpu_seconds[gpu_type] = 0
                        job_gpu_seconds[gpu_type] += len(gpu_list) * SCHED_INTERVAL
                    else:
                        queued += 1
                    gpu_seconds[job_entry["name"]] = job_gpu_seconds
            log_entry["gpu_seconds"] = gpu_seconds
            log_entry["running"] = running
            log_entry["queued"] = queued
            logs.append(log_entry)
        num_restarts = {x['name'] : x['num_restarts'] for x in entry['submitted_jobs']}
        # print(f"Num restarts: {num_restarts}")
    return logs, num_restarts


def read_logs_dir(dir):
    logs, num_restarts = {}, {}
    for logfile in glob.glob(dir + "/w*.log"):
        wname = os.path.basename(logfile)[:-4]
        logs[wname], num_restarts[wname] = read_logs(logfile)
    return logs, num_restarts


parser = argparse.ArgumentParser()
parser.add_argument("--workload_dir", type=str, help="path to workload dir")
parser.add_argument("--output_dir", type=str, help="path to logs dir")
parser.add_argument(
    "--interval", default=60, type=int, help="sched interval (default=60)"
)

args = parser.parse_args()

# set script params
workload_dir = args.workload_dir
output_dir = args.output_dir
SCHED_INTERVAL = args.interval

# read all workloads
workloads = {}
for fname in glob.glob(workload_dir + "/*.csv"):
    wname = os.path.basename(fname)[:-4]
    workloads[wname] = pd.read_csv(fname)
num_workloads = len(workloads)
print(f"Found {num_workloads} workloads")

# read summary
summary_file = os.path.join(output_dir, "summary.json")
with open(summary_file, "r") as f:
    summary = json.load(f)
jcts = summary["jcts"]
assert (
    len(jcts) == num_workloads
), f"got only {len(jcts)} summaries, expected {num_workloads}"


def get_makespans(workloads, jcts):
    makespans = []
    for wname, workload in workloads.items():
        jct_dict = jcts[wname]
        min_subtime = 100000000
        max_comptime = 0
        for i, row in workload.iterrows():
            subtime = row["time"]
            comp_time = subtime + jct_dict[row["name"]]
            min_subtime = min(min_subtime, subtime)
            max_comptime = max(max_comptime, comp_time)
        makespans.append(max_comptime - min_subtime)
    return makespans


def get_avg_jcts(jcts):
    all_jcts = []
    avg_jcts = []
    for wname, wjcts in jcts.items():
        all_jcts.extend(list(wjcts.values()))
        avg_jcts.append(np.mean(list(wjcts.values())))
    return np.mean(all_jcts), np.std(avg_jcts)


def get_p99_jct(jcts):
    jct_vals = []
    for wname, wjcts in jcts.items():
        jct_vals.extend(wjcts.values())
    jct_vals = sorted(jct_vals)
    idx = int(0.99 * len(jct_vals))
    return jct_vals[idx]


def mean(list_vals):
    return sum(list_vals) * 1.0 / len(list_vals)


makespans = get_makespans(workloads, jcts)
avg_jcts, std_jcts = get_avg_jcts(jcts)
p99_jct = get_p99_jct(jcts)
print(
    f"Avg Makespan (hrs) (over {num_workloads} workloads) = {mean(makespans)/3600.0}, range = {min(makespans)/3600.0, max(makespans)/3600.0}"
)
print(
    f"Avg JCT (hrs) (over {num_workloads} workloads) = {avg_jcts/3600.0}, std = {std_jcts/3600.0}, p99 JCT = {p99_jct/3600.0}"
)

# read logs
sim_logs, sim_num_restarts = read_logs_dir(output_dir)

# print num restarts stats
def print_restarts_stats(num_restarts):
    total_num_restarts = sum([sum(x.values()) for x in num_restarts.values()])
    total_num_jobs = sum([len(x) for x in num_restarts.values()])
    print(f"Average num restarts per job = {total_num_restarts/total_num_jobs:.3f}")
    job_types = set()
    for _, wrestarts in num_restarts.items():
        for jname in wrestarts.keys():
            job_types.add(jname.split("-")[0])
    job_type_restarts = {jt: 0 for jt in sorted(job_types)}
    job_type_counts = {jt: 0 for jt in sorted(job_types)}
    for _, wrestarts in num_restarts.items():
        for jname, num_restarts in wrestarts.items():
            job_type = jname.split("-")[0]
            job_type_restarts[job_type] += num_restarts
            job_type_counts[job_type] += 1
    for jt, num_restarts in job_type_restarts.items():
        job_type_restarts[jt] = num_restarts / job_type_counts[jt]
    print(f"Average num restarts per job type: {job_type_restarts}")
print_restarts_stats(sim_num_restarts)

# print contention stats
def get_contention_stats(logs):
    stats = {}
    for wname, wlogs in logs.items():
        cont_vals = [v["running"] + v["queued"] for v in wlogs]
        mean_val = mean(cont_vals)
        max_val = max(cont_vals)
        stats[wname] = (mean_val, max_val)
    mean_all = mean([v[0] for v in stats.values()])
    max_all = max([v[1] for v in stats.values()])
    return stats, (mean_all, max_all)


cont_stats, (mean_cont, max_cont) = get_contention_stats(sim_logs)
print(f"Contention: mean={mean_cont}, max={max_cont}")


# print queue stats
def get_queue_stats(logs):
    stats = {}
    all_q_vals = []
    for wname, wlogs in logs.items():
        q_vals = [v["queued"] for v in wlogs]
        all_q_vals.extend(q_vals)
        mean_val = mean(q_vals)
        max_val = max(q_vals)
        stats[wname] = (mean_val, max_val)
    mean_all = mean([v[0] for v in stats.values()])
    median_all = np.median(all_q_vals)
    max_all = max([v[1] for v in stats.values()])
    return stats, (mean_all, median_all, max_all)


q_stats, (mean_q, median_q, max_q) = get_queue_stats(sim_logs)
print(f"queue length: mean={mean_q}, median={median_q}, max={max_q}")


# print GPU hrs stats
def get_gpu_hrs_stats(logs):
    avg_gpu_hrs = {}
    counts = {}
    # each workload
    for wname, wlogs in logs.items():
        all_jobnames = set()
        # each timestamp log
        for entry in wlogs:
            # each active job
            for job, job_gpu_seconds in entry["gpu_seconds"].items():
                app = job.split("-")[0]
                if app not in avg_gpu_hrs:
                    avg_gpu_hrs[app] = {}
                    counts[app] = 0
                if job not in all_jobnames:
                    all_jobnames.add(job)
                    counts[app] += 1
                for gpu_type, gpu_hrs in job_gpu_seconds.items():
                    if gpu_type not in avg_gpu_hrs[app]:
                        avg_gpu_hrs[app][gpu_type] = 0
                    avg_gpu_hrs[app][gpu_type] += gpu_hrs / 3600.0
    num_jobs = sum(counts.values())
    mean_hrs = sum([sum(v.values()) for v in avg_gpu_hrs.values()]) / num_jobs
    print(f"Job counts: {counts}")

    for app in avg_gpu_hrs.keys():
        for gpu_type in avg_gpu_hrs[app].keys():
            avg_gpu_hrs[app][gpu_type] /= counts[app]
    return mean_hrs, avg_gpu_hrs


mean_gpu_hrs, avg_gpu_hrs = get_gpu_hrs_stats(sim_logs)
print(f"GPU hours: mean={mean_gpu_hrs}, per-job={avg_gpu_hrs}")


def get_ftf_ratios(jcts, ftf_jcts):
    fratios = {}
    for jname, jct in jcts.items():
        if ftf_jcts[jname] == 0:
            print(f"{jname}")
        fratios[jname] = jct / ftf_jcts[jname]
    return fratios


ftf_summary = None
FTF_SUMMARY_FILE = os.path.join(output_dir, "ftf_summary_true_share.json")
if os.path.exists(FTF_SUMMARY_FILE):
    with open(FTF_SUMMARY_FILE, "r") as f:
        ftf_summary = json.load(f)
    ftf_jcts = ftf_summary["jcts"]
    # compute FTF ratios for each job in each workload
    ftf_ratios = {}
    for wname, wjcts in jcts.items():
        ftf_ratios[wname] = get_ftf_ratios(wjcts, ftf_jcts[wname])
    ftf_ratios_list = []
    for wname, wratios in ftf_ratios.items():
        ftf_ratios_list.extend(list(wratios.values()))
    ftf_ratios_list = np.asarray(ftf_ratios_list)
    # print FTF stats
    unfair_ratio = sum(ftf_ratios_list > 1) / sum(ftf_ratios_list > 0)
    max_ftf_ratio = np.max(ftf_ratios_list)
    perc_vals = [0, 50, 90, 95, 99, 100]
    ftf_percentiles = np.percentile(ftf_ratios_list, perc_vals)
    print(
        f"FTF:: unfair_ratio={unfair_ratio:.4f}, max_ftf_ratio={max_ftf_ratio:.4f}, ftf_percentiles={list(zip(perc_vals, ftf_percentiles))}"
    )
