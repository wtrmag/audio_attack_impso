import sys
import math
import argparse
import tensorflow as tf
import numpy as np
from eval import Eval, load_wav
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
    def __init__(self, input_wav_file, phrase, capacity=100, iterations=3000, c1=0.5, c2=0.5):
        self.w_ini = 0.9
        self.w_end = 0.1
        self.c1, self.c2 = c1, c2
        self.shift = 100
        self.noise_stedv = 40
        self.swarm_capacity = capacity
        self.max_iterations = iterations
        self.audio = load_wav(input_wav_file)
        self.dimension = len(self.audio)
        self.flying_speed = init_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
        self.eval = Eval(self.audio, phrase, capacity)
        self.gbest = self.pbest = self.swarm = self.eval.input_audio

    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        # region_extreme_index = [scores[i:i + shift].argmin() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        # region_extreme = [scores[i:i+shift].min() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def run(self):
        evaluate = self.eval
        r1 = np.random.rand(self.swarm_capacity, self.dimension)
        r2 = np.random.rand(self.swarm_capacity, self.dimension)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            pre_scores, curr_text, pre_mfcc = evaluate.get_fitness(sess, self.swarm)
        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores, self.shift)
        gbest = [self.swarm[ind].tolist() for ind in pre_index]

        itr = 1
        while itr <= self.max_iterations and curr_text != self.eval.target_phrase:
            sess = tf.Session()
            tf.global_variables_initializer()

            # dist = levenshtein_distance(curr_text, self.eval.target_phrase)
            print('***** ITERATION {} *****'.format(itr))
            self.swarm = self.swarm + self.flying_speed
            scores, curr_text, mfcc = evaluate.get_fitness(sess, self.swarm)
            # scores, curr_text = evaluate.get_fitness(sess, self.swarm + self.flying_speed)
            for k in range(self.swarm_capacity):
                if scores[k] > pre_scores[k]:
                    # if scores[k] < pre_scores[k]:
                    self.pbest[k] = self.swarm[k]
                    pre_scores[k] = scores[k]

            curr_best_score, curr_index = self.find_region_extreme(scores, self.shift)
            for i in range(self.swarm_capacity // self.shift):
                if curr_best_score[i] > pre_gbest_score[i]:
                    # if curr_best_score[i] < pre_gbest_score[i]:
                    gbest[i] = self.swarm[curr_index[i]].tolist()
                    pre_index[i] = curr_index[i]
                    pre_gbest_score[i] = curr_best_score[i]
            temp = np.stack([gbest] * self.shift, axis=1)
            self.gbest = np.vstack(temp)

            print('Current phrase: {}'.format(curr_text))
            print('Current best_score: {}'.format(max(pre_gbest_score)))

            # w = (self.w_ini - self.w_end) * (self.max_iterations - itr) / self.max_iterations + self.w_end
            # w = self.w_ini * (self.w_ini - self.w_end) * (self.max_iterations - itr) / self.max_iterations
            # w = self.w_ini - (self.w_ini - self.w_end) * math.pow((itr / self.max_iterations), 2)
            # w = math.pow(self.w_end * (self.w_ini / self.w_end), 1 / (1 + 10 * itr / self.max_iterations))
            self.flying_speed = self.flying_speed + self.c1 * r1 * (self.pbest - self.swarm) \
                                + self.c2 * r2 * (self.gbest - self.swarm)

            itr += 1

        return curr_text == self.eval.target_phrase


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, dest='input_wav_file', required=True)
    parser.add_argument('-t', '--target', type=str, dest='phrase', required=True)

    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    input_wav_file = args.input_wav_file
    phrase = args.phrase.lower()
    out_wav_file = input_wav_file[:-4] + '_adv.wav'
    log_file = input_wav_file[:-4] + '_log.txt'

    print('target phrase:', phrase)
    print('source file:', input_wav_file)

    pso = Particle_Swarm(input_wav_file, phrase)
    res = pso.run()
    print('success' if res else 'fail')
