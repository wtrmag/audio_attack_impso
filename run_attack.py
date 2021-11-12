import os
import sys
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter
from util.etxt import ctc_label_dense_to_sparse
import click
import neat
import tensorflow as tf
from tf_neat.multi_env_eval import MultiEnvEvaluator
from tf_neat.neat_reporter import LogReporter
from tf_neat.recurrent_net import RecurrentNet
from neat.nn.feed_forward import FeedForwardNetwork
from tf_logits import get_logits


# Activate eager TensorFlow execution
# tf.enable_eager_execution()
# print("Executing eagerly: ", tf.executing_eagerly())
toks = " abcdefghijklmnopqrstuvwxyz'-"


def db(audio):
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])


def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    assert fs == 16000
    print('source dB', db(audio))
    return audio


def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2 ** 15, 2 ** 15 - 1), dtype=np.int16))
    print('output dB', db(audio))


def levenshteinDistance(s1, s2):
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


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


def make_net(genome, config, bs):
    return RecurrentNet.create(genome, config, bs)


def activate_net(net, states):
    outputs = net.activate(states).numpy()
    return outputs[:, 0] > 0.5


def getctcloss(self, input_audio_batch, target_phrase, decode=False):
    batch_size = input_audio_batch.shape[0]
    weird = (input_audio_batch.shape[1] - 1) // 320
    logits_arg2 = np.tile(weird, batch_size)
    dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
    dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

    pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
    seq_len = np.tile(weird, batch_size).astype(np.int32)

    logits = get_logits(pass_in, logits_arg2)
    target = ctc_label_dense_to_sparse(dense_arg1, dense_arg2, batch_size)
    ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=seq_len)
    decoded, _ = tf.nn.ctc_greedy_decoder(logits, logits_arg2, merge_repeated=True)

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, "models/session_dump")
    if decode:
        return sess.run([ctcloss, decoded])
    else:
        return sess.run(ctcloss)


def get_fitness_score(self, input_audio_batch, target_phrase, input_audio, classify=False):
    target_enc = np.array([toks.index(x) for x in target_phrase])
    if classify:
        ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
        all_text = "".join([toks[x] for x in decoded[0].values])
        index = len(all_text) // input_audio_batch.shape[0]
        final_text = all_text[:index]
    else:
        ctcloss = self.getctcloss(input_audio_batch, target_enc)
    score = -ctcloss
    if classify:
        return (score, final_text)
    return score, -ctcloss


@click.command()
@click.option("--n_generations", type=int, default=100)
@click.option("--path", type=str, default="sample_input.wav")
@click.option("--target", type=str, default="Hello World")
def run(n_generations, path, target):
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    config_path = os.path.join(os.path.dirname(__file__), "neat.cfg")
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

    def eval_genomes(genomes, config):
        audio = load_wav(path)
        for genome_id, genome in genomes:
            net = RecurrentNet.create(genome, config)
            pop_scores, ctc = get_fitness_score(pop, target, audio)
            elite_ind = np.argsort(pop_scores)[-10:]
            elite_pop, elite_pop_scores, elite_ctc = pop[elite_ind], pop_scores[elite_ind], ctc[elite_ind]
            # genome.fitness =


    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    reporter = neat.StdOutReporter(True)
    pop.add_reporter(reporter)
    # logger = LogReporter("neat.log", evaluator.eval_genome)
    # pop.add_reporter(logger)

    pop.run(eval_genomes, n_generations)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter