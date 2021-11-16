import sys
import argparse
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

sys.path.append('DeepSpeech')
import DeepSpeech

tokens = " abcdefghijklmnopqrstuvwxyz'-"


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


class Attack():
    def __init__(self, input_wav_file, output_wav_file, phrase, batch_size=1, iterations=3000):
        self.audio = load_wav(input_wav_file)
        self.input_audio = np.tile(self.audio, (batch_size, 1))
        self.output_audio = output_wav_file
        self.target_phrase = phrase
        self.batch_size = batch_size
        self.audio_length = np.tile(len(self.audio), batch_size)
        self.target_index = np.tile(np.array([[tokens.index(x) for x in phrase]]), (batch_size, 1))
        self.target_length = np.tile(len(phrase), batch_size)
        self.weird = np.tile((self.audio.shape[0] - 1) // 320, 1)

        self.iterations = iterations

    def get_loss(self, sess):
        input = tf.placeholder(tf.float32, shape=self.input_audio.shape, name='input')
        input_len = tf.placeholder(tf.int32, shape=self.audio_length.shape, name='length')
        target_in = tf.placeholder(tf.int32, shape=self.target_index.shape, name='index')
        target_len = tf.placeholder(tf.int32, shape=self.target_length.shape, name='len')
        arg = tf.placeholder(tf.int32, shape=self.weird.shape, name='arg')

        pass_in = tf.clip_by_value(input, -2 ** 15, 2 ** 15 - 1)
        logits = get_logits(pass_in, arg)
        target = ctc_label_dense_to_sparse(target_in, target_len)
        ctc_loss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits,
                                  sequence_length=arg)
        decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg, merge_repeated=True)
        # decoded, _ = tf.nn.ctc_beam_search_decoder(logits, arg, merge_repeated=False, beam_width=100)

        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, "models/session_dump")

        ctc_loss, decode = sess.run([ctc_loss, decoded], feed_dict={input: self.input_audio, input_len: self.audio_length, target_in:
            self.target_index, target_len: self.target_length, arg: self.weird})
        print('ctc loss :{}'.format(ctc_loss))
        new_text = ''.join([tokens[i] for i in decode[0].values])
        print(new_text)


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

    attack = Attack(input_wav_file, out_wav_file, phrase)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        attack.get_loss(sess)



