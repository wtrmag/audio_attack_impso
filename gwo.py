import numpy as np


class Grey_Wolf():
    def __init__(self, eval, dimension, a=2):
        self.a = a
        self.swarm_capacity = eval.batch_size
        self.dimension = dimension
        self.swarm = eval.input_audio
        self.alpha = self.beta = self.delta = self.swarm[0, :]

    def hunting(self, a, leader, swarm):
        r1 = np.random.rand(self.swarm_capacity, 1)
        r2 = np.random.rand(self.swarm_capacity, 1)
        l_ = np.tile(leader, (self.swarm_capacity, 1))
        a_ = 2 * a * r1 - a
        c_ = 2 * r2

        d_ = c_ * l_ - swarm
        dist = np.reshape(np.sum(d_**2, axis=1), (self.swarm_capacity, 1))
        x_ = l_ - a_ * dist
        return d_, x_

    def process(self, swarm, scores, max_iter):
        index = np.argsort(scores)[self.swarm_capacity - 3:]
        self.alpha = swarm[index[2]]
        self.beta = swarm[index[1]]
        self.delta = swarm[index[0]]

        D_alpha, X1 = self.hunting(self.a, self.alpha, swarm)
        D_beta, X2 = self.hunting(self.a, self.beta, swarm)
        D_delta, X3 = self.hunting(self.a, self.delta, swarm)

        swarm = (X1 + X2 + X3) / 3

        self.a = self.a - (2 / max_iter)
        return swarm, [1] * sb












