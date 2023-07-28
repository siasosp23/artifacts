# Sia simulator
Supports simulating heterogeneous clusters (even larger-than-physical clusters) and builds on Pollux's OSDI 2021 artifact release. This artifact makes following additional contributions:
- Support for heterogeneous clusters
- Support for hybrid-parallel jobs
- Support for rigid jobs with varying level of rigidity
- Support for Gavel and Shockwave schedulers (Shockwave is currently disabled as it requires a license for Gurobi solver)


## Pre-requisites:
- Python 3.10+
- `pip install -r requirements.txt`
- `pip install cvxpy[CBC, GLPK]`

## Running simulator on traces
See `./run_experiment.sh` for examples on how to run the simulator for Sia/Gavel on Saturn traces.