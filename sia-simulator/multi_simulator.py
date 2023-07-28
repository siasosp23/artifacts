import argparse
import copy
import glob
import json
import multiprocessing
import os
import pandas
from rich.console import Console
from rich.table import Table

from cluster import Cluster
from gavel import GavelPolicy
from sia import SiaPolicy
from sia_fix import SiaFixPolicy
import simulator_config as sim_config


def multi_simulate(args):
    workload = pandas.read_csv(args.workload)
    if args.cluster_scale is not None:
        for k in sim_config.cluster_nnodes.keys():
            sim_config.cluster_nnodes[k] = sim_config.cluster_nnodes[k] * args.cluster_scale
    cluster_populate_nnodes = sim_config.cluster_nnodes

    # filter out GPU types not in simulated cluster
    sim_ngpus_per_node = {k: sim_config.cluster_ngpus_per_node[k] for k in sim_config.cluster_nnodes.keys()}

    if args.policy == "gavel":
        policy = GavelPolicy(args.interval, policy=sim_config.gavel_default_policy)
    elif args.policy == "sia_fix":
        policy = policy = SiaFixPolicy(p_fairness=args.policy_p_val,
                                               restart_penalty=30,
                                               lambda_a=args.mip_lambda_a,
                                               lambda_n=args.mip_lambda_n,
                                               project_throughputs=args.project_throughputs,
                                               share_max_replicas=args.share_max_replicas)
    else:
        policy = SiaPolicy(p_fairness=args.policy_p_val,
                                   restart_penalty=30,
                                   lambda_a=args.mip_lambda_a,
                                   lambda_n=args.mip_lambda_n,
                                   project_throughputs=args.project_throughputs,
                                   share_max_replicas=args.share_max_replicas,
                                   enable_bsz_tuning=not args.disable_bsz_tuning)
    policy.populate_valid_configs(
        cluster_populate_nnodes, sim_ngpus_per_node, sim_config.cluster_max_physical_nnodes)
    cluster = Cluster(workload, policy, sim_config.cluster_nnodes, sim_ngpus_per_node,
                      max_physical_nodes=sim_config.cluster_max_physical_nnodes)
    if args.disable_bsz_tuning and args.policy == "sia":
        print(f"Diabled bsz tuning for {args.policy} policy")
        cluster.disable_bsz_tuning()
    if args.policy == "gavel":
        cluster.disable_bsz_tuning()
    if args.policy == "sia":
        # seed profiles for all rigid jobs
        rigid_jobs = [job for job in cluster.jobs if job.category == "rigid"]
        for job in rigid_jobs:
            job.seed_profiles_rigid(sim_ngpus_per_node)

    if args.oracle_num_nodes != 0:
        for job in cluster.jobs:
            job.seed_profiles(args.oracle_num_nodes, args.oracle_num_replicas)

    current_time = 0
    while not cluster.all_complete():
        ####### STEP #######
        cluster.step(args.interval)
        table = Table(title=f"SIMULATOR TIME:{cluster.current_time}")
        table.add_column("Jobname", justify="left", style="cyan", no_wrap=True)
        table.add_column("Epoch", justify="left", style="white", no_wrap=True)
        table.add_column("Restarts", justify="left", style="white", no_wrap=True)
        table.add_column("Batch size", justify="left", style="white", no_wrap=True)
        table.add_column("Placement", justify="left", style="white", no_wrap=True)
        table.add_column("Contention", justify="left", style="white", no_wrap=True)
        table.add_column("Cluster", justify="left", style="white", no_wrap=True)
        table.add_column("Category", justify="left", style="white", no_wrap=True)

        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] > cluster.current_time or val["completion_time"] is not None:
                continue
            table.add_row(val["name"], str(val["epoch"]), str(val["num_restarts"]),
                          str(val["batch_size"]), str(val["placement"]), str(val["n_avg"]),
                          str(val["cluster"]), str(val["category"]))
        console = Console()
        console.print(table)
        '''
        print("Active jobs:")
        print("  Jobname\t\tEpoch\tRestarts\tBatch size\tPlacement\t\tContention\tCluster \tCategory".expandtabs(8))
        print("  =======\t\t=====\t========\t==========\t=========\t\t==========\t========\t========".expandtabs(8))
        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= cluster.current_time and val["completion_time"] is None:
                print(f"  {str(val['name']):<20}\t{str(val['epoch']):<4}\t{str(val['num_restarts']):<8}\t{str(val['batch_size']):<8}\t{str(val['placement']):<8}\t\t{str(val['n_avg'])[:5]:<8}\t{str(val['cluster']):<8}\t{str(val['category']):<8}".expandtabs(8))
        '''
        used_gpus = get_used_gpus(cluster.logs[-1], cluster.current_time)
        print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        jct_dict = cluster.get_jcts()
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) /
              len(jct_dict) if jct_dict else 0)

        if args.debug:
            inp = input(f'\n\nPress any key to simlate next round.... (x to quit)')
            if inp == "x":
                print(f"Quitting...")
                exit(0)
            elif inp == "c":
                print(f"Continuing...")
                args.debug = False

    if args.output:
        cluster.output_logs(args.output)
    return cluster.logs, cluster.get_jcts()


def get_used_gpus(log_entry, current_time):
    used_gpus = dict()
    for val in log_entry["submitted_jobs"]:
        if val["submission_time"] <= current_time and val["completion_time"] is None:
            cluster = val["cluster"]
            if cluster is None:
                continue
            else:
                if cluster not in used_gpus:
                    used_gpus[cluster] = 0
                used_gpus[cluster] += sum(val["placement"])
    return used_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="sia",
                        choices=["sia", "gavel", "sia_fix"])
    parser.add_argument("--policy_p_val", type=float,
                        default=0.5, help="value of p for policy=sia")
    parser.add_argument("--mip_lambda_n", type=float, default=None,
                        help="sia regularization: no-alloc")
    parser.add_argument("--mip_lambda_a", type=float, default=None,
                        help="sia regularization: change of alloc")
    parser.add_argument("--cluster_scale", type=int, default=None,
                        help="scale of cluster relative to hardcoded values")
    parser.add_argument("--project_throughputs", action='store_true', default=False,
                        help="projects throughput functions from one cluster to another by multiplying by a constant = ratio of mean of throughputs for num_replicas=1")
    parser.add_argument("--disable_bsz_tuning", action='store_true', default=False,
                        help="use supplied bsz from workload, scale only num GPUs (sia only)")
    parser.add_argument("--share_max_replicas", action='store_true',
                        default=False, help="share the maximum number of profiled replicas across clusters")
    parser.add_argument("--oracle_num_nodes", type=int, default=0,
                        help="max-num-nodes to seed profiles for, in each cluster")
    parser.add_argument("--oracle_num_replicas", type=int,
                        default=0, help="number of replicas to seed profiles for")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--output", type=str, help="path to output logs")
    parser.add_argument("--debug", action='store_true', default=False,
                        help="DEBUG MODE: simulate a scheduling round, pause and wait for user input to simulate next round")

    args = parser.parse_args()
    if sim_config.num_parallel_jobs > 1:
        assert sim_config.num_parallel_traces == 1, f"num_parallel_jobs={sim_config.num_parallel_jobs} so, num_parallel_traces must be 1 (check simulator_config.py)"
    if sim_config.num_parallel_traces > 1:
        assert sim_config.num_parallel_jobs == 1, f"num_parallel_traces={sim_config.num_parallel_traces} so, num_parallel_jobs must be 1 (check simulator_config.py)"
    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        if sim_config.num_parallel_traces > 1:
            with multiprocessing.Pool(processes=sim_config.num_parallel_traces) as pool:
                ret_list = pool.map(multi_simulate, args_list)
        else:
            ret_list = []
            for args_item in args_list:
                ret_list.append(multi_simulate(args_item))
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        multi_simulate(args)