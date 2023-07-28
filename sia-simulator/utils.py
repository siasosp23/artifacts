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

# Modified: Suhas Jayaram Subramanya (suhasjs@cs.cmu.edu)

class JobInfo(object):
    def __init__(self, resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, multipliers=dict(),
                 category=None, scale_unit=1):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
            category (str) : [adaptive, rigid, scale-gpus]
            scale_unit (int) : number of GPUs to scale up or down by (default: 1)
        """
        for k, v in max_replicas.items():
            assert v > 0, f"max_replicas = 0 for cluster: {k}"
        for k in min_replicas.keys():
            assert min_replicas[k] <= max_replicas[k], f"min/max-replicas invalid"
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.multipliers = multipliers
        self.category = category


class JobInfoUnaware(object):
    def __init__(self, resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, multipliers=dict(),
                 category=None, scale_unit=1):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
            category (str) : [adaptive, rigid, scale-gpus]
            scale_unit (int) : number of GPUs to scale up or down by (default: 1)
        """
        assert max_replicas > 0, f"max_replicas = 0"
        assert min_replicas <= max_replicas, f"min/max-replicas invalid"
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.multipliers = multipliers
        self.scale_unit = 1


class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible
