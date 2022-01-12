import numpy as np
from scipy.signal import butter, lfilter


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def init_noise(swarm_capacity, dimension, noise_stedv):
    noise = np.random.randn(swarm_capacity, dimension) * noise_stedv
    b, a = butter(10, 0.75, btype='high', analog=False)
    return lfilter(b, a, noise)


class Particle_Swarm():
    def __init__(self, eval, dimension):
        self.w_ini = 0.9
        self.w_end = 0.1
        self.c1, self.c2 = 0.5, 0.5
        self.shift = 100
        self.noise_stedv = 40
        self.swarm_capacity = eval.batch_size
        self.dimension = dimension
        self.gbest = self.pbest = self.swarm = eval.input_audio
        self.flying_speed = init_noise(eval.batch_size, dimension, self.noise_stedv)

    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def process(self, swarm, pre_scores, scores, max_iter, itr):
        r1 = np.random.rand(self.swarm_capacity, 1)
        r2 = np.random.rand(self.swarm_capacity, 1)

        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores, self.shift)
        gbest = [swarm[ind].tolist() for ind in pre_index]

        for k in range(self.swarm_capacity):
            if scores[k] > pre_scores[k]:
                self.pbest[k] = swarm[k]
                pre_scores[k] = scores[k]

        curr_best_score, curr_index = self.find_region_extreme(scores, self.shift)
        for i in range(self.swarm_capacity // self.shift):
            if curr_best_score[i] > pre_gbest_score[i]:
                gbest[i] = swarm[curr_index[i]].tolist()
                pre_index[i] = curr_index[i]
                pre_gbest_score[i] = curr_best_score[i]
        temp = np.stack([gbest] * self.shift, axis=1)
        self.gbest = np.vstack(temp)

        w = (self.w_ini - self.w_end) * (max_iter - itr) / max_iter + self.w_end
        self.flying_speed = w * self.flying_speed + self.c1 * r1 * (self.pbest - swarm) \
                            + self.c2 * r2 * (self.gbest - swarm)
        swarm = swarm + self.flying_speed

