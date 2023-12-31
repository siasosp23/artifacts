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


import json
import logging
import requests
from collections import OrderedDict
from adaptdl.goodput import PerfParams
import adaptdl.env
from types import MappingProxyType


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

NODE_TO_CLUSTER_MAP = MappingProxyType({
  "phodgx1" : "dgx",
  "phodgx2" : "dgx",
  "phortx1" : "rtx",
  "phortx2" : "rtx",
  "phortx3" : "rtx",
  "phoquad1" : "quad",
})

# make immutable proxies of globals
PERF_PARAMS = MappingProxyType(
    OrderedDict({k: 0.0 for k in PerfParams._fields}))

SCHED_HINTS = MappingProxyType({'initBatchSize': 0,
                                'localBszBounds': None,  # [min, max]
                                'globalBatchSize': None,
                                'maxBatchSize': 0,
                                'maxProfiledReplicas': 0,
                                'gradientAccumulation': False,
                                'epoch': None,
                                'batchSize': None,
                                'gradParams': None,
                                'perfParams': None,
                                'nodeName' : None, # node reporting hints
                                'localBszBoundsDict' : None, # script reporting hints for GPU types
                                'perfParamsDict' : None, # perf params for each GPU type
                                'perfParamsHintDict' : None}) # perf params hints for each GPU type, offline profiling

def post_sched_hints(sched_hints, job_key):
    url = adaptdl.env.supervisor_url()
    if not url or url == "":
        return  # skip
    headers = {"Content-Type": "application/json"}
    try:
        for k in sched_hints:
            assert k in SCHED_HINTS  # validate
        LOG.info(f"Posting hints: {sched_hints}")
        response = \
            requests.put(url=f"{url}/hints/{job_key}",
                         data=json.dumps(sched_hints),
                         headers=headers)
        if response.status_code != 200:
            LOG.warning(f"Received {response.status_code}")
    except Exception as e:
        LOG.warning(f"{e}")
