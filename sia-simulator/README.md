# Sia simulator
Supports simulating heterogeneous clusters (even larger-than-physical clusters) and builds on Pollux's OSDI 2021 artifact release. This artifact makes following additional contributions:
- Support for heterogeneous clusters
- Support for hybrid-parallel jobs
- Support for rigid jobs with varying level of rigidity
- Support for Gavel and Shockwave schedulers (Shockwave is currently disabled as it requires a license for Gurobi solver)


## Before you run simulator, do the following:
- Create new conda env with python 3.10: `conda create -n opt python=3.10`
- Install numpy+mkl from anaconda channel: `conda install numpy=1.23 -c anaconda`
- Install pandas+scipy from anaconda channel: `conda install pandas scipy -c anaconda`
- Install pymoo (to run Pollux policy): `pip install -U pymoo==0.4.2.1`
- Install cvxpy with CBC,GLPK solver support (to run Sia and Gavel policy): `pip install cvxpy[CBC, GLPK]`

## Running simulator on traces
See `./run_experiment.sh` for examples on how to run the simulator for Sia/Gavel/Pollux on Saturn traces.

## Summarizing results
Use `python utils/print_run_stats.py --workload_dir=${WORKLOAD_DIR} --output_dir=${OUTPUT_DIR} --interval=${INTERVAL}` to print summary of results for a run of simulator. This script outputs all relevant scheduler metrics from the run onto STDOUT in a human readble form (see below for a sample output).
```
root@phortx1:/artifacts/sia-simulator# python utils/print_run_stats.py --workload_dir=workloads/saturn/ --output_dir=/tmp/gavel_saturn --interval=360
Found 10 workloads
Avg Makespan (hrs) (over 10 workloads) = 43.25152777777778, range = (27.21361111111111, 59.57472222222222)
Avg JCT (hrs) (over 10 workloads) = 2.548240277777778, std = 0.9456597205311155, p99 JCT = 46.333888888888886
Reading /tmp/gavel_saturn/workload-7.log
Reading /tmp/gavel_saturn/workload-4.log
Reading /tmp/gavel_saturn/workload-9.log
Reading /tmp/gavel_saturn/workload-2.log
Reading /tmp/gavel_saturn/workload-1.log
Reading /tmp/gavel_saturn/workload-10.log
Reading /tmp/gavel_saturn/workload-8.log
Reading /tmp/gavel_saturn/workload-3.log
Reading /tmp/gavel_saturn/workload-6.log
Reading /tmp/gavel_saturn/workload-5.log
Average num restarts per job = 7.408
Average num restarts per job type: {'bert': 0.9142857142857143, 'cifar10': 0.7907153729071538, 'deepspeech2': 4.96, 'imagenet': 204.93478260869566, 'yolov3': 11.814285714285715}
Contention: mean=9.245206956348978, max=48
queue length: mean=4.992874877258684, median=2.0, max=39
Job counts: {'cifar10': 1314, 'yolov3': 70, 'deepspeech2': 100, 'bert': 70, 'imagenet': 46}
GPU hours: mean=12.00549999999987, per-job={'cifar10': {'dgx-ext': 0.7167427701674292, 'aws': 0.37990867579908666, 'rtx': 0.9372907153729089}, 'yolov3': {'aws': 15.891428571428271, 'rtx': 1.3942857142857186, 'dgx-ext': 1.3885714285714328}, 'deepspeech2': {'dgx-ext': 0.726, 'rtx': 3.6600000000000215, 'aws': 3.6679999999999886}, 'bert': {'rtx': 0.5971428571428575, 'dgx-ext': 1.0457142857142856, 'aws': 0.7771428571428574}, 'imagenet': {'rtx': 119.09999999999684, 'aws': 97.31739130434764, 'dgx-ext': 93.4565217391296}}
```
