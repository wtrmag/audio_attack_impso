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


def noise_mask(noise, direction_vector):
    assert noise.shape == direction_vector.shape
    x = (noise >= 0).astype(np.int32)
    y = (direction_vector >= 0).astype(np.int32)
    c = (x - y) == 0
    d = (1 - c).astype(np.bool)
    noise[d] = 0


class Particle_Swarm():
    def __init__(self, input_wav_file, phrase, capacity=100, iterations=3000, c1=0.5, c2=0.5):
        self.w_ini = 0.9
        self.w_end = 0.1
        self.c1, self.c2 = c1, c2
        self.shift = 10
        self.noise_stedv = 40
        self.swarm_capacity = capacity
        self.max_iterations = iterations
        self.audio = load_wav(input_wav_file)
        self.dimension = len(self.audio)
        self.flying_speed = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
        self.eval = Eval(self.audio, phrase, capacity)
        self.gbest = self.pbest = self.swarm = self.eval.input_audio

    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def run(self, iterations=50):
        tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True)
        evaluate = self.eval
        r1 = np.random.rand(self.swarm_capacity, self.dimension)
        r2 = np.random.rand(self.swarm_capacity, self.dimension)
        with tf.Session() as sess:
            pre_scores, curr_text = evaluate.get_fitness(sess, self.swarm)
        best_phrase = curr_text[0]
        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores, self.shift)
        gbest = [self.swarm[ind].tolist() for ind in pre_index]

        itr = 1
        flag = [1] * (self.swarm_capacity // self.shift)
        phrase_index = 0
        max_var_noise = self.flying_speed
        print('----- PERIOD    {} -----'.format(itr // 50 + 1))
        while itr <= iterations and best_phrase != evaluate.target_phrase:
            sess = tf.Session(config=config)

            print('***** ITERATION {} *****'.format(itr))
            temp_swarm = self.swarm + self.flying_speed
            scores, curr_text = evaluate.get_fitness(sess, temp_swarm)
            best_phrase = curr_text[phrase_index]
            curr_gbest_score, curr_index = self.find_region_extreme(scores, self.shift)
            pre_noise = self.flying_speed

            r = Counter(flag)
            if itr % 50 != 0:
                if r[1] > r[0] or itr == 2:
                    print('Current phrase: {}'.format(best_phrase))
                    print('Current best_score: {}'.format(max(pre_gbest_score)))
                    for k in range(self.swarm_capacity):
                        if scores[k] > pre_scores[k]:
                            self.pbest[k] = temp_swarm[k]
                            pre_scores[k] = scores[k]
                            max_var_noise[k] = self.flying_speed[k]

                    for i in range(self.swarm_capacity // self.shift):
                        if curr_gbest_score[i] > pre_gbest_score[i]:
                            gbest[i] = temp_swarm[curr_index[i]].tolist()
                            pre_index[i] = curr_index[i]
                            pre_gbest_score[i] = curr_gbest_score[i]
                            flag[i] = 1
                        else:
                            flag[i] = 0
                    temp = np.stack([gbest] * self.shift, axis=1)
                    self.gbest = np.vstack(temp)

                    self.swarm = temp_swarm

                    if itr % 10 == 0:
                        tf.reset_default_graph()
                        if evaluate.target_phrase in curr_text:
                            break
                    self.flying_speed = self.flying_speed + self.c1 * r1 * (self.pbest - self.swarm) \
                                        + self.c2 * r2 * (self.gbest - self.swarm)
                else:
                    print('update iteration')
                    dists = [get_levenshtein_distance(text, evaluate.target_phrase) for text in curr_text]
                    min_index = np.argsort(dists)[:self.shift]
                    phrase_index = min_index[0]

                    t = (self.flying_speed - pre_noise)
                    n = [t[o] for o in min_index]
                    new_noise = np.tile(np.mean(n, axis=0), (self.swarm_capacity, 1))

                    w = math.pow(self.w_end * (self.w_ini / self.w_end), 1 / (1 + 10 * itr / self.max_iterations))
                    v = 0.8 * new_noise + 0.2 * max_var_noise
                    noise_mask(v, max_var_noise)
                    self.swarm = w * self.swarm + v

                    flag = [1] * (self.swarm_capacity // self.shift)
            else:
                print('----- PERIOD    {} -----'.format(itr // 50 + 1))
                random_noise = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
                noise_mask(random_noise, max_var_noise)
                self.flying_speed = random_noise

            itr += 1
            sess.close()

        return itr, best_phrase, curr_text


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
    itr, try_phrase, all_text = pso.run(iterations=pso.max_iterations)
    e = time.time()
    if itr < pso.max_iterations or (itr == pso.max_iterations and phrase in all_text):
        ii = all_text.index(phrase)
        save_wav(pso.swarm[ii], out_wav_file)
        print("adversarial example successfully generates with {} s".format((e - s) / 1000))
    else:
        print("we try it,but still fail")
        print("best result after {} iterations: {}".format(pso.max_iterations, try_phrase))