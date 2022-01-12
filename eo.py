import numpy as np
import math

class Equilibrium():
    def __init__(self, eval, dimension):
        self.GP = 0.5
        self.a1, self.a2 = 1, 1
        self.swarm_capacity = eval.batch_size
        self.dimension = dimension
        self.swarm = eval.input_audio
        self.C_eq = self.swarm[:4].copy()
        self.fit_eq = np.array([float('-inf')] * 4)

    def process(self, swarm, scores, max_iter, itr, sb):
        oldsw = swarm.copy()
        C_ind = np.argsort(scores)[self.swarm_capacity - 4:]
        i = 0
        j = 0
        sc = np.array([float('-inf')] * 4)
        ceq = self.swarm[:4].copy()
        for k in range(4):
            if scores[C_ind[i]] > self.fit_eq[j]:
                sc[k] = scores[C_ind[i]]
                ceq[k] = self.swarm[C_ind[i]]
                i += 1
            else:
                sc[k] = self.fit_eq[j]
                ceq[k] = self.C_eq[j]
                j += 1
        self.C_eq = ceq
        self.fit_eq = sc

        C_ave = np.average(self.C_eq, axis=0)
        C_eq_pool = np.append(self.C_eq, np.expand_dims(C_ave, axis=0), axis=0)

        t = np.float_power(1 - itr / max_iter, (itr / max_iter) * self.a2)
        C_eq = C_eq_pool[np.random.randint(0, 5)]

        r = np.random.rand(self.swarm_capacity, 1)
        l = np.random.rand(self.swarm_capacity, 1)

        F = self.a1 * np.sign(r - 0.5) * (np.exp(-l * t) - 1)
        if np.random.rand() > self.GP:
            GCP = 0.5 * np.random.rand(self.swarm_capacity, 1)
        else:
            GCP = np.zeros((self.swarm_capacity, 1))
        G0 = GCP * (np.tile(C_eq, (self.swarm_capacity, 1)) - l * swarm)
        G = G0 * F
        swarm = C_eq + (swarm - C_eq) * F + G / (l * math.log(2999 - itr)/math.log(1000)) * (1 - F)

        return swarm, [1] * sb