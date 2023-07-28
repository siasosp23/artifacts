# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time

from collections import Counter
from datetime import datetime, timedelta
from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl_sched.policy.unaware_pollux import UnawarePolluxPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
from adaptdl_sched.cluster_config import get_mock_phoebe_node

# using perf params for 4-GPUs cifar10 on phodgx1 (~2m into training)
def test_optimize():
  # Make up a realistic speedup function.
  '''
  perf_params = PerfParams(0.121, 0.00568, 0.0236, 0.00634,
               0.0118, 0.00317, 1.14)
  '''
  perf_params = PerfParams(0.023, 9.08e-5, 0.0133, 0.008036,
                           0.012109, 0.007306, 1.914794)
  grad_params = GradParams(sqr=0.10985, var=2.94965)
  goodput_fn = GoodputFunction(perf_params, grad_params, 128)
  speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=4096,
                 atomic_bsz_range = (32, 4096), accumulation=True)
  now=datetime.now()
  jobs={}
  # Add a few jobs.
  job_resources={'pods': 1, 'cpu': 9, 'memory': 40, 'nvidia.com/gpu': 1, 'rdma/hca': 1}
  for i in range(2):
    creation_timestamp=now,
    max_replicas=8
    min_replicas=0
    key="cifar10-"+str(i)
    jobs[key]=JobInfo(job_resources, speedup_fn, creation_timestamp,
              min_replicas, max_replicas)
    jobs[key].attained_service = 547.4
    jobs[key].num_restarts = 2
    jobs[key].age = 273

  # Add a few nodes == phodgx1
  node_resources = {'cpu': 255, 
           'memory': 1081,
           'nvidia.com/gpu': 4, 
           'pods': 84,
           'rdma/hca': 4}
  node_template = NodeInfo(node_resources, preemptible=False)
  nodes = {"phortx1": get_mock_phoebe_node("phortx1", node_template),
          "phoquad1" : get_mock_phoebe_node("phoquad1", node_template)}
  # Add a node template.
  policy = UnawarePolluxPolicy()
  prev_allocs = {}
  for i in range(3):
    start = time.time()
    allocations, desired_nodes = \
      policy.optimize(jobs, nodes, prev_allocs, node_template)
    print(f"Allocations: {allocations}")
    duration = time.time() - start
    print("optimize {}x ({}s sec):".format(i + 1, duration))
    node_count = Counter()
    for job_key, placement in allocations.items():
      assert len(placement) <= jobs[job_key].max_replicas
      for node_key in placement:
        node_count[node_key] += 1
    for node_key, count in node_count.items():
      assert count <= nodes[node_key].resources["nvidia.com/gpu"]
      assert count <= nodes[node_key].resources["pods"]

if __name__ == "__main__":
  test_optimize()