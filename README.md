# Artifacts for SOSP 2023
Our artifacts build on Pollux OSDI 2021's artifact release. Supplied artifacts contain scripts to reproduce simulator-based and physical-cluster based experiments. For simulator-based experiments, use code in `sia-simulator`.
## Simulator experiments
Scripts to reproduce results in Table 3 and instructions for the simulator can be found inside sia-simulator/README.md
**Note** 
## Physical cluster experiments
We implementation of Sia using AdaptDL. Specifically, we make **minimal** changes to [Pollux's OSDI 2021 artifact release](https://github.com/petuum/adaptdl/tree/osdi21-artifact) in the following files to support GPU heterogeneity using adaptdl's mechanisms to communicate *scheduler hints* from client-side that are then picked up by the Sia scheduler to realize heterogeneity-aware goodput-optimized clusterscheduling.
Sia scheduler is referred to as `mip` in the codebase for physical cluster experiments.
- **run_workload.py**: adds support for Sia scheduler through a temporary name 'mip'
- **adaptdl/adaptdl/**: added support for multiple GPU types in `env.py, scheduler_hints.py, _metrics.py, torch/data.py`. Also added support for per GPU-type batch-size limits in `torch/data.py` (data-loader for adaptdl client)
- **adaptdl/adaptdl/torch/seed/**: 1-GPU profiles for each job on different GPU types used to generate the batch-size limits and initial bootstrap throughput models in Sia
- **benchmark/clear_state.sh**: added ability to clear adaptdl state (including checkpoints). **Note:** you'll have to modify this script to point to the k8s PVC that stores adaptDL state.
- **benchmark/models/*/Dockerfile**: Upgraded dockers to pytorch 1.9+cu11.1 to support A100s. Also upgraded ubuntu to 20.04 from 18.04
- **benchmark/models/*/*.py**: Modified launch scripts to hint app type (cifar10, imagenet, etc.). Only the `mip`/Sia scheduler uses these hints and other schedulers (pollux, gavel) ignore them

### How to reproduce physical cluster results?
- Pick a scheduler (say `gavel`). Modify `sched/adaptdl_sched/allocator.py` to set `POLICY` variable to the scheduler name. Set scheduler round duration using `_sched_trigger_interval` in `AdaptDLAllocator` class.
- Run `make deploy` to deploy the scheduler and adaptdl server to k8s
- `cd benchmark` and run `python run_workload.py POLICY phoebe_workload.csv` to start the job-submission script
- In another terminal, run `python run_monitor.py --output=JSON_LOG_FILE --verbose` to log job and scheduler state every 15 seconds. We use this `JSON_LOG_FILE` to generate the plots in the paper.
- Once the job submission script finishes, you can run `kubectl get adaptdljob` to see if all jobs successfully completed (last column should say `Successful` and not `Failed/Waiting`)
- Average JCT is given by taking average of runtimes from JSON log file (you can also look at kubectl get adaptdljob to see individual runtimes), and makespan is the time taken for all jobs to complete

### Potential changes needed to run on your cluster
- **adaptdl/run_placement.py**: This generates profiles for all jobs used in this artifact. You'll need to modify this depending on cluster size and relevant placements that need profiled (probably only have to deal with number of GPUs per node and number of nodes really).
- **adaptdl/adaptdl/scheduler_hints.py**: `NODE_TO_CLUSTER_MAP` dictates mapping between nodenames and GPU types. You'll need to modify this to point to the correct mapping
- **benchmark/models/*/Dockerfile**: Change the docker image to point to your docker registry
- **benchmark/models/yolov3/config/yolov3_config_voc.py**: Change the `DATA_PATH` to point to your dataset path
- **benchmark/models/yolov3/utils/datasets.py**: Check `img_path` and ensure that it matches your dataset path
- **benchmark/pvc.cephfs.yaml**: Configure PVC as required. We use a local flash network-attached storage (`hot` storage class in our k8s cluster)
- **benchmark/run_workload.py**: Set `--repository` to point to your docker repository and check any mounts that you might need to change (look for `mounts.append` in the script)
- **sched/adaptdl_sched/cluster_config.py**: modify cluster configuration to reflect your cluster's GPU types and their counts, along with node names and mapping of node names to node IDs
- **sched/adaptdl_sched/policy/unaware_pollux.py**: If your cluster contains different GPU types and different numbers of GPUs per node for each GPU type, you'll have to follow the template in `optimize()` under `DEBUG_PHOEBE` to homogenize your cluster so that Unaware-Pollux only sees one number of GPUs per node (agnostic to GPU type anyway). This is not required for Sia as it can handle heterogeneity natively.
