import sys
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse
from tf_logits import get_logits

# sys.path.append('DeepSpeech')
# import DeepSpeech

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


class Eval():
    def __init__(self, audio, phrase, batch_size=1):
        self.input_audio = np.tile(audio, (batch_size, 1))
        self.target_phrase = phrase
        self.batch_size = batch_size
        self.audio_length = np.tile(len(audio), batch_size)
        self.target_index = np.tile(np.array([[tokens.index(x) for x in phrase]]), (batch_size, 1))
        self.target_length = np.tile(len(phrase), batch_size)
        self.weird = np.tile((audio.shape[0] - 1) // 320, batch_size)

    def get_loss(self, sess, data):
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            inp = tf.placeholder(tf.float32, shape=self.input_audio.shape, name='input')
            input_len = tf.placeholder(tf.int32, shape=self.audio_length.shape, name='length')
            target_in = tf.placeholder(tf.int32, shape=self.target_index.shape, name='index')
            target_len = tf.placeholder(tf.int32, shape=self.target_length.shape, name='len')
            arg = tf.placeholder(tf.int32, shape=self.weird.shape, name='arg')

            pass_in = tf.clip_by_value(inp, -2 ** 15, 2 ** 15 - 1)
            logits = get_logits(pass_in, arg)
            target = ctc_label_dense_to_sparse(target_in, target_len)
            ctc_loss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits,
                                      sequence_length=arg)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg, merge_repeated=True)
            # decoded, _ = tf.nn.ctc_beam_search_decoder(logits, arg, merge_repeated=False, beam_width=100)

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, "models/session_dump")

        ctc_loss, decode = sess.run([ctc_loss, decoded],
                                    feed_dict={inp: data, input_len: self.audio_length, target_in:
                                        self.target_index, target_len: self.target_length, arg: self.weird})
        # print('ctc loss :{}'.format(ctc_loss))
        return ctc_loss, decode

    def get_fitness(self, sess, data):
        loss, decode = self.get_loss(sess, data)
        all_text = ''.join([tokens[i] for i in decode[0].values])
        index = len(all_text) // self.batch_size
        final_text = all_text[:index]
        return np.array(-loss), final_text

    def decode_text(self, decode):
        final_index = []
        str_index = []
        temp = 0
        ind = 0
        for i, j in decode[0].indices:
            if temp == i:
                str_index.append(decode[0].values[ind])
            else:
                final_index.append(str_index)
                temp += 1
                str_index = []
                str_index.append(decode[0].values[ind])
            ind += 1
        return [''.join([tokens[i] for i in index]) for index in final_index]
