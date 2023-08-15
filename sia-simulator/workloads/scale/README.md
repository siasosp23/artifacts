This folder contains workloads used to generate Fig.7 (scale job arrival rate at constant cluster size) and Fig. 9 (scale both job arrival rate and cluster size with same factor)

# Scale-rate
These workloads are sampled from Helios-Saturn traces using increasing _average_ job arrival rates (10 - 50 jobs/hr). `rate_30` corresponds to a set of 10 randomly sampled traces, each with an average job arrival rate of `30` jobs per hour.
## How to run?
We use `rate_*` workloads to obtain average JCT for Sia, Pollux (heterogeneity-unaware) and Gavel, so invoke the simulator with `rate_*` folder as the workload path and use `utils/print_run_stats.py` to print average JCT for the run.

# Scale-size
These workloads are constructed using Helios-Saturn traces (`../saturn`). `scale_8x` corresponds to a workload with 8x average job arrival rate meant to run on a cluster with 8x the evaluated _Heterogeneous_ cluster size. Consider `scale_8x/workload-1.csv`: each job (say `cifar10-0,401.0,cifar10,12,4096`) is issued `8` times to the scheduler with different job names (`cifar10-1, cifar10-2, ..., cifar10-7`) with the same hyper-parameters (`num_gups=12, bsz=4096`) at the same submission time (`t=401.0` seconds from start of trace). 
## How to run?
We use `scale_*x/workload-1.csv` to get scheduler runtimes at different cluster sizes. To simulate a cluster of size `8x` (`1x` = 64-GPU _Heterogeneous_ cluster used for main experiments), append `--cluster_scale=8` to command line when invoking the simulator. Here's a sample command line to simulate Gavel on a cluster with `8x` size.

`python multi_simulator.py workloads/scale/scale_8x/workload-1.csv --policy=gavel --interval=360 --cluster_scale=8 1>/tmp/gavel_8x_stdout.log 2>&1`

The above command also pipes `stdout,stderr` to a file `/tmp/gavel_8x_stdout.log`. Then, run `cat /tmp/gavel_8x_stdout.log | grep 'Policy optimization' | cut -f 1 -d ',' | cut -f 4 -d ' ' | sed 's/ms//g' > /tmp/gavel_8x_runtime.txt`.

The file `/tmp/gavel_8x_runtime.txt` now contains runtimes for each timestep. You can then use `cat /tmp/gavel_8x_runtime.txt | datamash mean 1 sstdev 1 perc:25 1 median 1 perc:75 1 perc:95 1`. This will print out `mean, stddev, 25th percentile, 50th percentile (median), 75th percentile, 95th percentile` using the utility `datamash` (you can install it using `sudo apt install datamash`.
