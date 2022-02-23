import sys
import math
import time
from collections import Counter
import argparse
import numpy as np
from eval import Eval, load_wav
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
import tensorflow as tf


def get_levenshtein_distance(s1, s2):
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


def rand_noise(swarm_capacity, dimension, noise_stedv):
    noise = np.random.randn(swarm_capacity, dimension) * noise_stedv
    b, a = butter(10, 0.75, btype='high', analog=False)
    return lfilter(b, a, noise)


def save_wav(audio, output_wav_file):
    x = np.array(np.clip(audio, -2 ** 15, 2 ** 15 - 1))
    wav.write(output_wav_file, 16000, x)


def noise_mask(noise, diff, cond):
    assert noise.shape == diff.shape
    if cond:
        e_ = np.sqrt(np.sum(np.square(diff), axis=0))
        direction_vector = diff / e_
        x = (noise >= 0).astype(np.int32)
        y = (direction_vector >= 0).astype(np.int32)
        c = (x - y) == 0
        d = (1 - c).astype(np.bool)
        noise[d] = 0
    else:
        q = np.argsort(np.abs(diff))[:, noise.shape[1] // 2:]
        for p in range(noise.shape[0]):
            noise[p, q] = 0

    return noise


class Particle_Swarm():
    def __init__(self, input_wav_file, phrase, capacity=500, iterations=3000, c1=0.9, c2=0.9):
        self.w_ini = 0.9
        self.w_end = 0.1
        self.c1, self.c2 = c1, c2
        self.shift = 500
        self.noise_stedv = 40
        self.swarm_capacity = capacity
        self.max_iterations = iterations
        self.audio = load_wav(input_wav_file)
        self.dimension = len(self.audio)
        self.flying_speed = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
        self.eval = Eval(self.audio, phrase, capacity)
        self.swarm = self.eval.input_audio.copy()
        self.pbest = self.eval.input_audio.copy()
        self.gbest = self.eval.input_audio.copy()

    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def run(self, iterations, ites=30):
        tf.global_variables_initializer()
        evaluate = self.eval
        r1 = np.random.rand(self.swarm_capacity, self.dimension)
        r2 = np.random.rand(self.swarm_capacity, self.dimension)
        with tf.Session() as sess:
            pre_scores, curr_text = evaluate.get_fitness(sess, self.swarm)
        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores.copy(), self.shift)
        gbest = [self.swarm[ind].copy() for ind in pre_index]

        lasts, iter = 0, 1
        flag = [1] * (self.swarm_capacity // self.shift)
        phrase_index = 0
        max_var_noise = self.flying_speed.copy()
        org_audio = evaluate.input_audio.copy()
        print('----- PERIOD    {} -----'.format(1))
        while iter <= iterations and evaluate.target_phrase not in curr_text:
            sess = tf.Session()

            temp_swarm = self.swarm + self.flying_speed
            scores, curr_text = evaluate.get_fitness(sess, temp_swarm)
            best_phrase = curr_text[phrase_index]
            curr_gbest_score, curr_index = self.find_region_extreme(scores.copy(), self.shift)

            flag.append(scores)
            loss = -scores.max()
            if iter % ites != 0:
                condition = np.var(flag[-10:], axis=0) > 10
                c = Counter(condition)
                print('***** ITERATION {} *****'.format(iter))
                if len(flag) < 10 or c[1] > c[0] or lasts < 5:
                    print('Current phrase: {}'.format(best_phrase))
                    print('Current best_score: {}'.format(loss))
                    for k in range(self.swarm_capacity):
                        if scores[k] > pre_scores[k]:
                            self.pbest[k] = temp_swarm.copy()[k]
                            pre_scores[k] = scores.copy()[k]
                            max_var_noise[k] = self.flying_speed.copy()[k]

                    for i in range(self.swarm_capacity // self.shift):
                        if curr_gbest_score[i] > pre_gbest_score[i]:
                            gbest[i] = temp_swarm.copy()[curr_index[i]]
                            pre_index[i] = curr_index.copy()[i]
                            pre_gbest_score[i] = curr_gbest_score.copy()[i]
                    temp = np.stack([gbest] * self.shift, axis=1)
                    self.gbest = np.vstack(temp)

                    self.swarm = temp_swarm
                    lasts = lasts + 1
                    if iter % 10 == 0:
                        tf.reset_default_graph()

                    self.flying_speed = self.flying_speed + self.c1 * r1 * (self.pbest - self.swarm) \
                                        + self.c2 * r2 * (self.gbest - self.swarm)

                else:
                    print('update iteration')
                    x = self.swarm_capacity // 10
                    dists = [get_levenshtein_distance(text, evaluate.target_phrase) for text in curr_text]
                    max_dist = np.argsort(dists)[-x:]
                    min_dist = np.argsort(dists)[:x]

                    u = np.arange(0, self.swarm_capacity, 1)
                    max_var = u[condition]
                    min_var = u[(condition == False)]

                    bad_intern = np.intersect1d(max_dist, min_var)

                    if len(bad_intern) != 0:
                        print('knock out!')
                        d = np.setdiff1d(u, bad_intern)
                        good_intern = np.intersect1d(min_dist, max_var)
                        if len(good_intern) >= len(bad_intern):
                            print('best backup')
                            self.swarm = np.append(self.swarm.copy()[d],
                                                   self.swarm.copy()[np.random.choice(good_intern, len(bad_intern),
                                                                                      replace=False)], axis=0)
                        else:
                            print('second backup')
                            self.swarm = np.append(self.swarm.copy()[d],
                                                   self.swarm.copy()[np.random.choice(min_dist, len(bad_intern),
                                                                                      replace=False)], axis=0)
                    lasts = 0
            else:
                print('----- PERIOD    {} -----'.format(iter // ites + 1))
                print('***** ITERATION {} *****'.format(iter))
                tt = self.swarm.copy()
                diff = tt - org_audio
                random_noise = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)

                T = np.var(flag, axis=0) > 100/3
                F = np.var(flag, axis=0) <= 100/3
                self.swarm[T] = self.swarm[T] + noise_mask(random_noise.copy(), diff, True)[T]
                self.swarm[F] = self.swarm[F] + noise_mask(random_noise.copy(), diff, False)[F]

                org_audio = tt
                flag.clear()
                print('Recombination')

            iter += 1
            sess.close()

        return iter, self.swarm, curr_text


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
    s = time.time()
    itr, swarm, all_text = pso.run(iterations=pso.max_iterations)
    e = time.time()
    if itr < pso.max_iterations or (itr == pso.max_iterations and phrase in all_text):
        ii = all_text.index(phrase)
        save_wav(swarm[ii], out_wav_file)
        print("adversarial example is successfully generated with {} s".format(e - s))
    else:
        print("we try it,but still fail")
