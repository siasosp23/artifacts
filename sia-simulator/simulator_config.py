#### Simulated cluster config ####
# number of GPUs per node for each GPU type
# "aws" : g4dn.12xlarge instance on AWS EC2
# "azure" : ND40rs_v2 instance on Azure
# "dgx" : DGX-A100 (8x NVLink A100 40GB) on-prem, limited profiled batch sizes
# "dgx-ext" : DGX-A100 (8x NVLink A100 40GB) on-prem, extended profiled batch sizes (should be default)
# "quad" : 4x Quadro RTX 6000 on-prem
# "rtx" : 8x RTX 2080Ti on-prem
cluster_ngpus_per_node = {"aws": 4, "azure" : 8, "dgx": 8, "dgx-ext": 8, "quad" : 4, "rtx": 8}
# number of nodes per GPU type (vary to change simulated cluster makeup)
cluster_nnodes = {"aws": 6, "dgx-ext": 2, "rtx": 3}       # default for experiments
# cluster_nnodes = {"aws": 16}                              # homogeneous cluster
# cluster_nnodes = {"rtx": 3, "dgx-ext": 2, "quad": 1}      # phoebe config
# max number of profiled physical nodes for each GPU type
cluster_max_physical_nnodes = {"azure": 5, "aws": 16, "dgx": 2, "dgx-ext": 2, "quad": 1, "rtx": 3}

##### Simulator default params #####
## multi-processing ##
## useful for large traces
# num traces to run in parallel
# WARNING:: overrides --num_parallel_jobs to 1 and uses one process per trace if > 1
num_parallel_traces = 4
# num parallel processes per trace
# WARNING:: overrides --num_parallel_traces to 1 and runs Cluster.step_jobs with a process pool if > 1
num_parallel_jobs = 1
# whether to preserve allocations for jobs that are currently in checkpoint-restore
preserve_ckpt_allocs = True

## calibration ##
# slowdown values for each job fitted to real-world data
SLOWDOWNS = {"cifar10": 1.8, "imagenet": 1.2,
             "yolov3": 1.2, "bert": 1.2, "deepspeech2": 1.2, 
             "ncf": 1.5, "gpt_pmp": 1.0}
calibrate_cluster = 'rtx'

# logging options
log_cluster_summary = True
log_cluster_verbose = False
log_sched_decisions = False

#### Sia params ####
### ENSURE:: 
# 1. sign(lambda_n) == sign(lambda_a) [same sign for penalty/gain]
# 2. sign(lambda_n) != sign(p) [penalty for p>0, gain for p<0]
# 3. abs(lambda_n) > 1.0 [penalty for no-alloc > min-gain-from-alloc==1.0]
# lambda_n: penalty for no-alloc
sia_default_lambda_n = 1.1
# lambda_a: penalty for change of alloc
sia_default_lambda_a = 0.01
# p: fairness parameter
sia_default_p_val = -0.5
# clip normalized goodput values to this if any larger
sia_goodput_clip_val = 3000.0
# solver params
# solver verbosity control
sia_solver_verbose = True
# choose solver: GLPK/CBC
sia_solver = "glpk"
# solver threading control (only for CBC solver)
sia_solver_num_threads = 8
# solver optimality control
sia_solver_thresh = 1e-4
# other controls
sia_log_goodputs = False
sia_log_problem = False
# uses realloc factor defined in Pollux paper
# Alternate formulation uses GPU-seconds instead of wall-clock seconds
sia_use_pollux_realloc_factor = True
# implementation idiosyncrasies: adaptdl client needs to 
# start before sending scheduler hints to scheduler
sia_read_throughputs_before_first_run = False

#### Gavel default params ####
# choices: ["max_sum_throughput_perf", "finish_time_fairness_perf"]
gavel_default_policy = "max_sum_throughput_perf"
