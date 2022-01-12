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


def noise_mask(noise, dt):
    assert noise.shape == dt.shape
    e_ = np.sqrt(np.sum(np.square(dt), axis=0))
    direction_vector = dt / e_
    x = (noise >= 0).astype(np.int32)
    y = (direction_vector >= 0).astype(np.int32)
    c = (x - y) == 0
    d = (1 - c).astype(np.bool)
    noise[d] = 0


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
        self.gbest = self.pbest = self.swarm = self.eval.input_audio

    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def run(self, iterations, its=20, phrase_index=0):
        tf.global_variables_initializer()
        evaluate = self.eval
        temp_swarm = self.swarm
        with tf.Session() as sess:
            pre_scores, curr_text = evaluate.get_fitness(sess, self.swarm)
        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores, self.shift)
        gbest = [self.swarm[ind].tolist() for ind in pre_index]

        itr, y_ = 1, 0.2
        flag = [1] * (self.swarm_capacity // self.shift)
        max_var_noise = self.flying_speed
        org_audio = evaluate.input_audio
        org_loss = -pre_scores.max()
        # li = [[0]*self.dimension]
        print('----- PERIOD    {} -----'.format(1))
        while itr <= iterations and evaluate.target_phrase not in curr_text:
            sess = tf.Session()
            r1 = np.random.rand(self.swarm_capacity, self.dimension)
            r2 = np.random.rand(self.swarm_capacity, self.dimension)

            print('***** ITERATION {} *****'.format(itr))
            temp_swarm = self.swarm + self.flying_speed
            scores, curr_text = evaluate.get_fitness(sess, temp_swarm)
            best_phrase = curr_text[phrase_index]
            curr_gbest_score, curr_index = self.find_region_extreme(scores, self.shift)
            pre_noise = self.flying_speed

            r = Counter(flag)
            loss = -scores.max()
            if itr % its != 0:
                if r[1] > r[0] or itr % its == 1:
                    print('Current best phrase: {}'.format(best_phrase))
                    print('Current best score: {}'.format(loss))
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
                    # c = scores.max() - pre_scores.max()
                    # if (loss < 30 and c > 0.2) or (loss < 20 and c > 0.1):
                    #     li.append(self.flying_speed)

                    if itr % 10 == 0:
                        tf.reset_default_graph()
                    self.flying_speed = self.flying_speed + self.c1 * r1 * (self.pbest - self.swarm) \
                                        + self.c2 * r2 * (self.gbest - self.swarm)
                else:
                    print('update iteration')
                    if loss >= 30:
                        dists = [get_levenshtein_distance(text, evaluate.target_phrase) for text in curr_text]
                        min_index = np.argsort(dists)[:self.shift]
                        # phrase_index = min_index[0]

                        t = (self.flying_speed - pre_noise)
                        n = [t[o] for o in min_index]
                        v = np.tile(np.mean(n, axis=0), (self.swarm_capacity, 1))

                        w1 = math.pow(self.w_end * (self.w_ini / self.w_end), 1 / (1 + 10 * itr / self.max_iterations))
                        new_noise = max_var_noise + v
                        self.swarm = w1 * self.swarm + (1 - w1) * new_noise
                    else:
                        dif = temp_swarm - self.swarm
                        # n_ = np.mean(li, axis=0)
                        q = np.argsort(np.abs(dif))[:, self.dimension - self.shift * 2:]
                        random_noise = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
                        for p in range(self.swarm_capacity):
                            random_noise[p, q] = 0
                        w2 = math.pow(self.w_end * (self.w_ini / self.w_end), 1 / (1 + 10 * itr / self.max_iterations))
                        self.swarm = w2 * self.swarm + (1 - w2) * random_noise
                        # li = [[0]*self.dimension]

                    flag = [1] * (self.swarm_capacity // self.shift)
            else:
                print('----- PERIOD    {} -----'.format(itr // its + 1))
                tt = self.swarm
                diff = tt - org_audio
                random_noise = rand_noise(self.swarm_capacity, self.dimension, self.noise_stedv)
                if org_loss - loss >= 2:
                    noise_mask(random_noise, diff)
                else:
                    q = np.argsort(np.abs(diff))[:, self.dimension // 2:]
                    for p in range(self.swarm_capacity):
                        random_noise[p, q] = 0
                self.flying_speed = random_noise
                if org_loss - loss < y_:
                    self.swarm = org_audio
                org_audio = tt
                org_loss = loss
                iii = np.argsort(scores)[self.swarm_capacity - self.shift:]
                self.swarm = np.tile(self.swarm[iii], (self.swarm_capacity // self.shift, 1))
                # if loss < 20:
                #     its, y_ = 10, 0.1

            itr += 1
            sess.close()

        return itr, temp_swarm, curr_text


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