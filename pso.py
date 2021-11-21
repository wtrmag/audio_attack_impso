import sys
import argparse
import tensorflow as tf
import numpy as np
from eval import Eval, load_wav
from scipy.signal import butter, lfilter


class Particle_Swarm():
    def __init__(self, input_wav_file, phrase, capacity=100, iterations=3000, c1=0.5, c2=0.5):
        self.w_ini = 0.9
        self.w_end = 0.4
        self.c1, self.c2 = c1, c2
        self.shift = 2
        self.noise_stedv = 40
        self.swarm_capacity = capacity
        self.max_iterations = iterations
        self.audio = load_wav(input_wav_file)
        self.dimension = len(self.audio)
        self.flying_speed = self.init_noise()
        self.eval = Eval(self.audio, phrase, capacity)
        self.gbest = self.pbest = self.swarm = self.eval.input_audio

    def init_noise(self):
        noise = np.random.randn(self.swarm_capacity, self.dimension) * self.noise_stedv
        b, a = butter(10, 0.75, btype='high', analog=False)
        return lfilter(b, a, noise)

    # def generate_noise(self, mutation_p):
    #     noise = np.random.randn() * self.noise_stedv
    #     b, a = butter(10, 0.75, btype='high', analog=False)
    #     noise = lfilter(b, a, noise)
    #     mask = np.random.rand(self.swarm_capacity, self.dimension) < mutation_p
    #     return noise * mask
    def find_region_extreme(self, scores, shift):
        region_extreme_index = [scores[i:i + shift].argmax() + i for i in range(0, scores.shape[0], shift)]
        region_extreme = [scores[i:i + shift].max() for i in range(0, scores.shape[0], shift)]
        # region_extreme = [scores[i:i+shift].min() for i in range(0, scores.shape[0], shift)]
        return region_extreme, region_extreme_index

    def run(self, sess):
        evaluate = self.eval
        r1 = np.random.rand(self.swarm_capacity, self.dimension)
        r2 = np.random.rand(self.swarm_capacity, self.dimension)
        pre_scores, curr_text = evaluate.get_fitness(sess, self.swarm)
        pre_gbest_score, pre_index = self.find_region_extreme(pre_scores, self.shift)
        gbest = [self.swarm[ind].tolist() for ind in pre_index]

        itr = 1
        while itr <= self.max_iterations and curr_text != self.eval.target_phrase:
            print('***** ITERATION {} *****'.format(itr))
            print('Current phrase: {}'.format(curr_text))
            self.swarm = self.swarm + self.flying_speed
            scores, curr_text = evaluate.get_fitness(sess, self.swarm)
            # scores, curr_text = evaluate.get_fitness(sess, self.swarm + self.flying_speed)
            for k in range(self.swarm_capacity):
                if scores[k] > pre_scores[k]:
                    self.pbest[k] = self.swarm[k]
                    pre_scores[k] = scores[k]
                # self.pbest[k] = self.swarm[k] + self.flying_speed[k] if scores[k] > pre_scores[k] else self.pbest[k]

            curr_best_score, curr_index = self.find_region_extreme(scores, self.shift)
            for i in range(self.swarm_capacity // self.shift):
                if curr_best_score[i] > pre_gbest_score[i]:
                    gbest[i] = self.swarm[curr_index[i]].tolist()
                    pre_index[i] = curr_index[i]
                    pre_gbest_score[i] = curr_best_score[i]
            temp = np.stack([gbest] * (self.swarm_capacity // self.shift), axis=1)
            self.gbest = np.vstack(temp)

            print('Current best_score: {}'.format(max(pre_gbest_score)))

            w = (self.w_ini - self.w_end) * (self.max_iterations - itr) / self.max_iterations + self.w_end
            self.flying_speed = w * self.flying_speed + self.c1 * r1 * (self.pbest - self.swarm) \
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
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        res = pso.run(sess)
        print('success' if res else 'fail')
