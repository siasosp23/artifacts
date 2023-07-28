from audioop import avg
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import cvxpy as cp
import numpy as np

from policy import Policy, PolicyWithPacking

class MinJCTPerf(Policy):

    def __init__(self, solver):
        Policy.__init__(self, solver)
        self._name = 'min_jct_perf'

    def get_allocation(self, unflattened_throughputs, scale_factors, cluster_spec, num_steps_remaining, age):
        throughputs, index = super().flatten(unflattened_throughputs,
                                             cluster_spec)
        if throughputs is None: return None
        (m, n) = throughputs.shape
        (job_ids, worker_types) = index


        num_steps = []
        time_elapsed = []

        for job_id in job_ids:
            num_steps.append(num_steps_remaining[job_id])
            time_elapsed.append(age[job_id])

        num_steps = np.asarray(num_steps).reshape(-1, 1)
        time_elapsed = np.asarray(time_elapsed).reshape(-1, 1)



        x = cp.Variable(throughputs.shape)
        avg_throughput = cp.reshape(cp.sum(cp.multiply(throughputs, x), axis = 1), (m, 1))
        jct = time_elapsed + cp.multiply(num_steps, cp.inv_pos(avg_throughput))
        objective = cp.Minimize(cp.sum(jct, axis=0))

        scale_factors_array = self.scale_factors_array(scale_factors, job_ids, m, n)
        constraints = self.get_base_constraints(x, scale_factors_array)

        cvxprob = cp.Problem(objective, constraints)
        result = cvxprob.solve(solver=self._solver)

        # print("\t--- passed jobs:", job_ids)
        # print("\t--- avg: throughput:", avg_throughput.value)
        # for i in range(len(job_ids)):
        #     print(f"\t---{job_ids[i]}, x: {x.value.round(decimals=5)[i, :]}, avg_throughput: {avg_throughput.value[i, :]}, jct:{jct.value[i, :]}")

        if cvxprob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')
            exit()

        return super().unflatten(x.value.round(decimals=5).clip(min=0.0).clip(max=1.0), index)