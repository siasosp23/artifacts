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

from re import L
import numpy as np
from goodput import GoodputFunctionPMP


class SpeedupFunction(object):
    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, mem_size=32, tune_bsz=True):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._mem_size = mem_size
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation, tune_bsz=tune_bsz)
        self.tune_bsz = tune_bsz
        # Memoization for fast repeated queries.
        self._mem_speedup = -np.ones((mem_size, mem_size))
        self._mem_speedup[0, 0] = 0.0
        # self._mem_speedup[1, 1] = 1.0

    def __call__(self, num_nodes, num_replicas):
        return self.get_goodput(num_nodes, num_replicas, True)

    def get_goodput(self, num_nodes, num_replicas, return_speedup_only=False):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))
        # Remember what the output shape/format should be and flatten inputs.
        output_scalar = np.isscalar(num_nodes) and np.isscalar(num_replicas)
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Return values which will be filled out.
        speedup = -np.ones(output_shape).flatten()
        # Fill in any previously memoized results first.
        indices = num_replicas < self._mem_size
        mem_idx = (num_nodes[indices], num_replicas[indices])
        speedup[indices] = self._mem_speedup[mem_idx]
        # Find the missing indices which still need to be computed.
        missing = speedup < 0
        if np.count_nonzero(missing) > 0:
            num_nodes, num_replicas = num_nodes[missing], num_replicas[missing]
            # Find unique inputs to reduce compuation.
            (num_nodes, num_replicas), inverse = np.unique(
                np.stack([num_nodes, num_replicas]),
                axis=1, return_inverse=True)
            goodput, _, _ = self._goodput_fn.optimize(
                num_nodes, num_replicas,
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._atomic_bsz_range,
                accumulation=self._accumulation,
                tune_bsz=self.tune_bsz)
            # Memoize results.
            indices = num_replicas < self._mem_size
            mem_idx = (num_nodes[indices], num_replicas[indices])
            self._mem_speedup[mem_idx] = goodput[indices] / self._base_goodput
            # Fill in computed results.
            speedup[missing] = goodput[inverse] / self._base_goodput
        assert np.all(np.less_equal(0, speedup))
        speedup = speedup.reshape(output_shape)
        speedup = speedup.item() if output_scalar else speedup
        if not return_speedup_only:
            goodput = speedup * self._base_goodput
            return goodput
        else:
            return speedup

    def set_base_goodput(self, baseline_num_nodes, baseline_num_replicas):
        self._base_goodput, _, _ = self._goodput_fn.optimize(
            num_nodes=baseline_num_nodes,
            num_replicas=baseline_num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation)


class UncachedSpeedupFunction(object):
    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, tune_bsz=True):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation, tune_bsz=tune_bsz)
        self.tune_bsz = tune_bsz
        # print(f"{self.__class__.__name__} --> bsz tuning: {self.tune_bsz}")

    def __call__(self, num_nodes, num_replicas):
        return self.get_goodput(num_nodes, num_replicas, True)

    # `num_nodes`, `num_replicas` MUST be numpy.ndarrays of type=np.float32/float64
    # YOU WILL GET WRONG RESULTS FOR dtype=np.uint32 for some reason`
    def get_goodput(self, num_nodes, num_replicas, return_speedup_only=False):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))

        goodputs, _, _ = self._goodput_fn.optimize(
            num_nodes, num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation,
            tune_bsz=self.tune_bsz)
        # print(f"goodputs: {goodputs}")
        if return_speedup_only:
            return goodputs / self._base_goodput
        else:
            return goodputs

    def set_base_goodput(self, baseline_num_nodes, baseline_num_replicas):
        # fast return for [1,1] baseline config
        if baseline_num_nodes == 1 and baseline_num_replicas == 1:
            return

        self._base_goodput, _, _ = self._goodput_fn.optimize(
            num_nodes=baseline_num_nodes,
            num_replicas=baseline_num_replicas,
            max_batch_size=self._max_batch_size,
            atomic_bsz_range=self._atomic_bsz_range,
            accumulation=self._accumulation)
