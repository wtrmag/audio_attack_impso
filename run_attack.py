import sys
import time
import argparse
from collections import Counter
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from pso import Particle_Swarm, generate_noise
from gwo import Grey_Wolf
from eo import Equilibrium
from eval import Eval, load_wav


def save_wav(audio, output_wav_file):
    x = np.array(np.clip(audio, -2 ** 15, 2 ** 15 - 1))
    wav.write(output_wav_file, 16000, x)


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


def masked_noise(noise, dt):
    assert noise.shape == dt.shape
    e_ = np.sqrt(np.sum(np.square(dt), axis=0))
    direction_vector = dt / e_
    x = (noise >= 0).astype(np.int32)
    y = (direction_vector >= 0).astype(np.int32)
    c = (x - y) == 0
    d = (1 - c).astype(np.bool)
    noise[d] = 0
    return noise


class Hybrid():
    def __init__(self, file, phrase, capacity, iterations):
        self.target_phrase = phrase
        self.swarm_capacity = capacity
        self.max_iterations = iterations
        self.audio = load_wav(file)
        self.dimension = len(self.audio)
        self.eval = Eval(self.audio, self.target_phrase, self.swarm_capacity)
        self.swarm = self.eval.input_audio

    def run(self, ites=10):
        evaluate = self.eval
        pso = Particle_Swarm(evaluate, self.dimension)
        # gwo = Grey_Wolf(evaluate, self.dimension)
        eo = Equilibrium(evaluate, self.dimension)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            pre_scores, curr_text = evaluate.get_fitness(sess, self.swarm)

        iter, iii = 1, 0
        sb = self.swarm_capacity // pso.shift
        flag = [1] * sb
        org_audio = evaluate.input_audio
        org_loss = -pre_scores.max()
        print('----- PERIOD    {} -----'.format(1))
        while iter <= self.max_iterations and self.target_phrase not in curr_text:
            sess = tf.Session()
            tf.global_variables_initializer()

            scores, curr_text = evaluate.get_fitness(sess, self.swarm)

            print('***** ITERATION {} *****'.format(iter))
            loss = -scores.max()
            print('Current phrase: {}'.format(curr_text[iii]))
            print('Current best_score: {}'.format(loss))
            r = Counter(flag)
            self.swarm, flag = eo.process(self.swarm, scores, self.max_iterations, iter, sb)
            # if iter % ites != 0:
            #     if r[1] > r[0]:
            #         print('pso iteration')
            #         self.swarm, flag = pso.process(self.swarm, pre_scores, scores, self.max_iterations, iter, flag)
            #     else:
            #         print('eo iteration')
            #         self.swarm, flag = eo.process(self.swarm, scores, self.max_iterations, iter, sb)
            # else:
            #     print('----- PERIOD    {} -----'.format(iter // ites + 1))
            #     tt = self.swarm
            #     diff = tt - org_audio
            #     random_noise = generate_noise(self.swarm_capacity, self.dimension, pso.noise_stedv)
            #     if org_loss - loss >= 2:
            #         random_noise = masked_noise(random_noise, diff)
            #     else:
            #         q = np.argsort(np.abs(diff))[:, self.dimension // 2:]
            #         for p in range(self.swarm_capacity):
            #             random_noise[p, q] = 0
            #     pso.flying_speed = random_noise
            #     # if org_loss - loss < 0.5:
            #     #     self.swarm = org_audio
            #     org_audio = tt
            #     org_loss = loss
            #     dists = [get_levenshtein_distance(text, evaluate.target_phrase) for text in curr_text]
            #     min_index = np.argsort(dists)[:pso.shift]
            #     iii = min_index[0]
            #     # iii = np.argsort(scores)[self.swarm_capacity - pso.shift:]
            #     self.swarm = np.tile(self.swarm[min_index], (sb, 1))

            iter += 1
            sess.close()
            if iter % 10 == 0:
                tf.reset_default_graph()
        return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, dest='input_wav_file', required=True)
    parser.add_argument('-t', '--target', type=str, dest='phrase', required=True)
    parser.add_argument('-c', '--cap', type=int, dest='itr', required=False, default=100)
    parser.add_argument('--it', type=int, dest='cap', required=False, default=3000)

    args = parser.parse_args()
    while len(sys.argv) > 1:
        sys.argv.pop()

    input_wav_file = args.input_wav_file
    phrase = args.phrase.lower()
    out_wav_file = input_wav_file[:-4] + '_adv.wav'

    print('target phrase:', phrase)
    print('source file:', input_wav_file)

    hybrid = Hybrid(input_wav_file, phrase, args.itr, args.cap)
    s = time.time()
    itr, swarm, all_text = hybrid.run()
    e = time.time()
    if itr < hybrid.max_iterations or (itr == hybrid.max_iterations and phrase in all_text):
        ii = all_text.index(phrase)
        save_wav(swarm[ii], out_wav_file)
        print("adversarial example is successfully generated with {} s".format(e - s))
    else:
        print("we try it,but still fail")