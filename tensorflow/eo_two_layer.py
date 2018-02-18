"""
Implements a 2-layer MNIST network for even/odd prediction which uses fisher information for combination.
"""
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from termcolor import colored

INPUT_SIZE = 784
OUTPUT_SIZE = 2

class MnistEO:

    def __init__(self, num_hidden, logger):
        """
        Initializes two layer network with num_hidden hidden units for even/odd task
        :param num_hidden: The number of neurons in the hidden layer
        """
        self.g = tf.Graph()
        self.logger = logger
        with self.g.as_default():
            self.sess = tf.Session()

            self.in_dim = int(INPUT_SIZE)
            self.out_dim = int(OUTPUT_SIZE)
            self.num_hidden = num_hidden

            # Create the model
            self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
            self.y_ = tf.placeholder(tf.float32, [None, self.out_dim])
            self.keep_prob = tf.placeholder(tf.float32)

            # simple 2-layer network
            self.W1 = weight_variable([self.in_dim, self.num_hidden])
            self.b1 = bias_variable([self.num_hidden])

            self.W2 = weight_variable([self.num_hidden, self.out_dim])
            self.b2 = bias_variable([self.out_dim])

            self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
            self.y = tf.matmul(self.h1, self.W2) + self.b2  # output layer

            self.var_list = [self.W1, self.b1, self.W2, self.b2]

            # vanilla single-task loss
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))

            # performance metrics
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
            self.sess.run(tf.global_variables_initializer())

    '''
       Trains the model on the provided train data. 
    '''
    def train_model(self, train, test, iters=2000):
        """
        Trains model and returns (accuracy, loss)
        :param train: training data
        :param test: test data
        :param iters: number of iterations
        :return: (accuracy, loss)
        """
        with self.g.as_default():
            for iter in range(iters):
                batch = train.next_batch(200)
                self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_: batch.labels})

            accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                           feed_dict={self.x: test.images, self.y_: test.labels})
            self.logger.info("Iters = %d Accuracy = %f Loss = %f" % (iters, accuracy, loss))
        return accuracy, loss

    def evaluate(self, data):
        """
        :param data: data to evaluate
        :return: (accuracy, loss)
        """
        with self.g.as_default():
            return self.sess.run([self.accuracy, self.cross_entropy],
                                 feed_dict={self.x: data.images, self.y_: data.labels})

    def get_w1(self):
        with self.g.as_default():
            return self.sess.run(self.W1)

    def get_w2(self):
        with self.g.as_default():
            return self.sess.run(self.W2)

    def get_b1(self):
        with self.g.as_default():
            return self.sess.run(self.b1)

    def get_b2(self):
        with self.g.as_default():
            return self.sess.run(self.b2)

    def get_fisher_w1(self):
        with self.g.as_default():
            return self.F_accum[0]

    def get_fisher_b1(self):
        with self.g.as_default():
            return self.F_accum[1]

    def get_fisher_w2(self):
        with self.g.as_default():
            return self.F_accum[2]

    def get_fisher_b2(self):
        with self.g.as_default():
            return self.F_accum[3]

    def set_vars(self, w1 = [], b1 = [], w2 = [], b2 = []):
        with self.g.as_default():
            if len(w1) > 0:
                self.sess.run(tf.assign(self.W1, w1))
            if len(b1) > 0:
                self.sess.run(tf.assign(self.b1, b1))
            if len(w2) > 0:
                self.sess.run(tf.assign(self.W2, w2))
            if len(b2) > 0:
                self.sess.run(tf.assign(self.b2, b2))

    def compute_fisher(self, images, num_samples=200):
        with self.g.as_default():
            self.logger.info("Computing Fisher...")
            # initialize Fisher information for most recent task
            self.F_accum = []
            for v in range(len(self.var_list)):
                self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

            # sampling a random class from softmax
            probs = tf.nn.softmax(self.y)
            class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

            for i in range(num_samples):
                # select random input image
                im_ind = np.random.randint(images.shape[0])
                # compute first-order derivatives
                ders = self.sess.run(tf.gradients(tf.log(probs[0, class_ind]), self.var_list),
                                feed_dict={self.x: images[im_ind:im_ind + 1]})
                # square the derivatives and add to total
                for v in range(len(self.F_accum)):
                    self.F_accum[v] += np.square(ders[v])

            # divide totals by number of samples
            for v in range(len(self.F_accum)):
                self.F_accum[v] /= num_samples
            self.logger.info("Fisher computed!")

    def copy(self):
        with self.g.as_default():
            new_m = MnistEO(self.num_hidden, self.logger)
            new_m.set_vars(w1=self.get_w1(), b1=self.get_b1(), w2=self.get_w2(), b2=self.get_b2())
            return new_m


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


class Solver():

    def __init__(self, m1, m2):

        self.sess = tf.Session()

        self.num_hidden = m1.num_hidden
        self.logger = m1.logger
        # Add w1 to loss
        m1_w1 = m1.get_w1()
        m2_w1 = m2.get_w1()
        m1_fw1 = m1.get_fisher_w1()
        m2_fw1 = m2.get_fisher_w1()

        self.w1 = tf.Variable(m1_w1, dtype=tf.float32)
        self.loss = tf.reduce_sum(tf.multiply(tf.abs(self.w1 - m1_w1), m1_fw1) + tf.multiply(tf.abs(self.w1 - m2_w1),
                                                                                         m2_fw1))

        # Add b1 to loss
        m1_b1 = m1.get_b1()
        m2_b1 = m2.get_b1()
        m1_fb1 = m1.get_fisher_b1()
        m2_fb1 = m2.get_fisher_b1()

        self.b1 = tf.Variable(m1_b1, dtype=tf.float32)
        self.loss = self.loss +  tf.reduce_sum(tf.multiply(tf.abs(self.b1 - m1_b1), m1_fb1) + \
                    tf.multiply(tf.abs(self.b1 - m2_b1), m2_fb1))

        # Add w2 to loss

        m1_w2 = m1.get_w2()
        m2_w2 = m2.get_w2()
        m1_fw2 = m1.get_fisher_w2()
        m2_fw2 = m2.get_fisher_w2()

        self.w2 = tf.Variable(m1_w2, dtype=tf.float32)
        self.loss = self.loss + tf.reduce_sum(tf.multiply(tf.abs(self.w2 - m1_w2), m1_fw2) + tf.multiply(tf.abs(
                self.w2 - m2_w2),m2_fw2))

        # Add b2 to loss

        m1_b2 = m1.get_b2()
        m2_b2 = m2.get_b2()
        m1_fb2 = m1.get_fisher_b2()
        m2_fb2 = m2.get_fisher_b2()

        self.b2 = tf.Variable(m1_b2, dtype=tf.float32)
        self.loss = self.loss + tf.reduce_sum(tf.multiply(tf.abs(self.b2 - m1_b2), m1_fb2) + \
                    tf.multiply(tf.abs(self.b2 - m2_b2), m2_fb2))


        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for i in range(20000):
            self.sess.run(self.train_step)
        self.logger.info("Iter-%d Solver Loss: %f" % (i, self.sess.run(self.loss)))

    def get_new_model(self):
        new_model = MnistEO(self.num_hidden, self.logger)
        new_model.set_vars(w1 = self.sess.run(self.w1),
                           b1 = self.sess.run(self.b1),
                           w2 = self.sess.run(self.w2),
                           b2 = self.sess.run(self.b2))
        return new_model
