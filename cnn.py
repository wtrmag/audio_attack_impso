import sys
import tensorflow as tf
import scipy.io.wavfile as wav
import numpy as np
from tf_logits import get_logits
from tensorflow.python.keras.backend import ctc_label_dense_to_sparse

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


#----Weight Initialization---#
#One should generally initialize weights with a small amount of noise for symmetry breaking, and to prevent 0 gradients
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#Convolution and Pooling
#Our convolutions uses a stride of one and are zero padded so that the output is the same size as the input.
#Our pooling is plain old max pooling over 2x2 blocks
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


class Attack():

    def __init__(self, input_wave_file, output_wave_file, target_phrase):
        self.pop_size = 100
        self.elite_size = 10
        self.mutation_p = 0.005
        self.noise_stdev = 40
        self.noise_threshold = 1
        self.mu = 0.9
        self.alpha = 0.001
        self.max_iters = 3000
        self.num_points_estimate = 100
        self.delta_for_gradient = 100
        self.delta_for_perturbation = 1e3
        self.input_audio = load_wav(input_wave_file).astype(np.float32)
        self.pop = np.expand_dims(self.input_audio, axis=0)
        self.pop = np.tile(self.pop, (self.pop_size, 1))
        self.output_wave_file = output_wave_file
        self.target_phrase = target_phrase


    def get_ctcloss(self):
        input_audio_batch = self.pop
        target_phrase = self.target_phrase

        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)

        pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)

        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
            len_batch = tf.placeholder(tf.float32, name='b')
            arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
            arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
            arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
            len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')

            logits = get_logits(inputs, arg2_logits)
            target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, "models/session_dump")

        func0 = lambda a, b, c, d, e, f: sess.run(ctcloss,
                                                  feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                             arg2_dense: e, len_seq: f})
        func1 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded],
                                                  feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d,
                                                             arg2_dense: e, len_seq: f})
        return (func0, func1)


    def run(self, decode):
        input_audio_batch = self.pop
        target_phrase = self.target_phrase
        target_in = np.array([toks.index(x) for x in target_phrase])

        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_in, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_in.shape[0], batch_size), dtype=np.int32)

        pass_in = np.clip(input_audio_batch, -2 ** 15, 2 ** 15 - 1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)

        if decode:
            self.get_ctcloss()[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
            # sess.run([ctcloss, decoded], feed_dict={inputs: pass_in, len_batch: batch_size, arg2_logits: logits_arg2,
            #                                         arg1_dense: dense_arg1, arg2_dense: dense_arg2, len_seq: seq_len})
        else:
            self.get_ctcloss()[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
            # sess.run(ctcloss, feed_dict={inputs: pass_in, len_batch: batch_size, arg2_logits: logits_arg2,
            #                              arg1_dense: dense_arg1, arg2_dense: dense_arg2, len_seq: seq_len})



if '__name__' == '__main__':
    inp_wav_file = sys.argv[1]
    target = sys.argv[2].lower()
    out_wav_file = inp_wav_file[:-4] + '_adv.wav'
    log_file = inp_wav_file[:-4] + '_log.txt'

    print('target phrase:', target)
    print('source file:', inp_wav_file)

    attack = Attack(inp_wav_file, out_wav_file, target)

    x = tf.placeholder(tf.float32)
    y_ = tf.placeholder(tf.float32)
    # ----first convolution layer----#
    # he convolution will compute 32 features for each 5x5 patch. Its weight tensor will have a shape of [5, 5, 1, 32].
    # The first two dimensions are the patch size,
    # the next is the number of input channels, and the last is the number of output channels.
    W_conv1 = weight_variable([5, 5, 1, 32])

    # We will also have a bias vector with a component for each output channel.
    b_conv1 = bias_variable([32])

    # We then convolve x_image with the weight tensor, add the bias, apply the ReLU function, and finally max pool.
    # The max_pool_2x2 method will reduce the image size to 14x14.
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # ----second convolution layer----#
    # The second layer will have 64 features for each 5x5 patch and input size 32.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # ----fully connected layer----#
    # Now that the image size has been reduced to 7x7, we add a fully-connected layer with 1024 neurons to allow processing on the entire image
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # -----dropout------#
    # To reduce overfitting, we will apply dropout before the readout layer.
    # We create a placeholder for the probability that a neuron's output is kept during dropout.
    # This allows us to turn dropout on during training, and turn it off during testing.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_dropout = tf.nn.dropout(h_fc1, keep_prob)

    # ----read out layer----#
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

    # ------train and evaluate----#
    fun = attack.get_ctcloss()
    train_step = tf.train.AdamOptimizer(100).minimize()
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_, 1), tf.argmax(y_conv, 1)), tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()




        print('the test accuracy :{}'.format())
        saver = tf.train.Saver(tf.global_variables())
        path = saver.restore(sess, "models/session_dump")
        print('save path: {}'.format(path))

