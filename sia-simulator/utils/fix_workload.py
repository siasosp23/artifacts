# code to resample high util configs for gavel
# author: Suhas Jayaram Subramanya (suhasj@cs.cmu.edu)

from applications import APPLICATIONS
import argparse
import pandas as pd
from dateutil import parser
import random

SAMPLE_CLUSTER = "rtx"
SAMPLE_NGPUS_PER_NODE = 8
SAMPLE_MAX_GPUS = 16
SAMPLE_SEED = 1

# set up argparse
parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, help="path to input workload csv")
parser.add_argument("--out_file", type=str, default="path to output workload csv")

args = parser.parse_args()
in_filename = args.in_file
out_filename = args.out_file

print(f"Reading workload from {in_filename}")
df = pd.read_csv(in_filename)
rng = random.Random(SAMPLE_SEED)

for i, row in df.iterrows():
  # fix num rows
  if row['num_replicas'] > SAMPLE_MAX_GPUS:
    app_name = row['application']
    app = APPLICATIONS[SAMPLE_CLUSTER][app_name]
    configs = app.get_configurations(lo_util=0.5, hi_util=0.8, max_gpus=SAMPLE_MAX_GPUS, ngpus_per_node=SAMPLE_NGPUS_PER_NODE)
    chosen_config = rng.choice(configs)
    print(f"Fixing: {i}, {row['name']}: {row['num_replicas']}, {row['batch_size']} --> {chosen_config[0]}, {chosen_config[1]}")
    df.loc[i, 'num_replicas'] = chosen_config[0]
    df.loc[i, 'batch_size'] = chosen_config[1]
  elif row['application'] == "ncf":
    # change to cifar10
    df.loc[i, 'name'] = "cifar10-"+str(i)
    df.loc[i, 'application'] = "cifar10"
    app = APPLICATIONS[SAMPLE_CLUSTER]["cifar10"]
    configs = app.get_configurations(lo_util=0.5, hi_util=0.8, max_gpus=SAMPLE_MAX_GPUS, ngpus_per_node=SAMPLE_NGPUS_PER_NODE)
    chosen_config = rng.choice(configs)
    print(f"Fixing: {i}, {row['name']}: {row['num_replicas']}, {row['batch_size']} --> {chosen_config[0]}, {chosen_config[1]}")
    df.loc[i, 'num_replicas'] = chosen_config[0]
    df.loc[i, 'batch_size'] = chosen_config[1]

print(f"Writing fixed workload to {out_filename}")
df.to_csv(out_filename, index=False)