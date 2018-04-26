"""
Implements a model for digit/letter learning
"""
import tensorflow as tf
import model_utils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from loss_calculator import *

INPUT_SIZE = 784
OUTPUT_SIZE = 10  # 10 digits
BETA = 0.01

''' 
We represent a neuron as (in_weights, out_weights), where in_weights corresponds to the incoming weights from the input
layer, and out_weights corresponds to the weights going to the output neurons. 
'''

class CombinedModel:

    def __init__(self):
        self.hidden = 0
        self.layer1_w = np.zeros((784, self.hidden)) # should have shape (784, num hidden)
        self.layer2_w = np.zeros((self.hidden, 10)) # should have shape (num hidden, 10)
        self.layer1_b = np.zeros(self.hidden)
        self.layer2_b = np.zeros(10)

    def add_hidden_neuron(self, in_weights, out_weights, bias):
        in_weights = np.resize(in_weights, (784, 1))
        out_weights = np.resize(out_weights, (1, 10))
        self.layer1_w = np.append(self.layer1_w, in_weights, axis=1)
        self.layer1_b = np.append(self.layer1_b, bias)
        self.layer2_w = np.append(self.layer2_w, out_weights, axis=0)
        self.hidden += 1

    def add_output_bias(self, output_index, bias):
        self.layer2_b[output_index] = bias


    def merge_models(self, m1, m2):
        models = [m1, m2]
        for m in models:
            w1 = m.get_w1()
            b1 = m.get_b1()
            w2 = m.get_w2()
            b2 = m.get_b2()
            for i in range(m.num_hidden):
                self.add_hidden_neuron(w1[:, i], w2[i, :], b1[i])
            self.layer2_b = self.layer2_b + b2
        self.layer2_b = self.layer2_b / 2.0
        return self.convert_to_model()


    def convert_to_model(self):
        model = FlexibleModel(self.hidden)
        model.set_vars(w1 = self.layer1_w, b1 = self.layer1_b, w2 = self.layer2_w, b2 = self.layer2_b)
        return model



class Agent:

    def __init__(self, num_hidden):

        self.input_size = 784
        self.output_size = 10
        self.model = FlexibleModel(num_hidden)

    def get_neuron(self, layer, i):
        if layer == 1:
            w = self.model.get_w1()[:, i]
            b = self.model.get_b1()[i]
        elif layer == 2:
            w = self.model.get_w2()[:, i]
            b = self.model.get_b2()[i]
        return (w, b)

    def set_neuron(self, layer, i, w, b):
        if layer == 1:
            weights = self.model.get_w1()
            weights[:, i] = w
            biases = self.model.get_b1()
            biases[i] = b
            self.model.set_vars(w1 = weights, b1 = biases)
        elif layer == 2:
            weights = self.model.get_w2()
            weights[:, i] = w
            biases = self.model.get_b2()
            biases[i] = b
            self.model.set_vars(w2 = weights, b2 = biases)


class FlexibleModel:

    def __init__(self, num_hidden):
        """
        Initializes two layer network with num_hidden hidden units for even/odd task
        :param num_hidden: The number of neurons in the hidden layer
        :param logger: logger to write to
        """
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.Session()

            self.in_dim = int(INPUT_SIZE)
            self.out_dim = int(OUTPUT_SIZE)
            self.num_hidden = num_hidden

            # Create the model
            # TODO: Add Regularizer
            self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
            self.y_ = tf.placeholder(tf.float32, [None, self.out_dim])
            self.keep_prob = tf.placeholder(tf.float32)

            # simple 2-layer network
            self.W1 = model_utils.weight_variable([self.in_dim, self.num_hidden])
            self.b1 = model_utils.bias_variable([self.num_hidden])

            self.W2 = model_utils.weight_variable([self.num_hidden, self.out_dim])
            self.b2 = model_utils.bias_variable([self.out_dim])

            self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
            self.h_fc1_drop = tf.nn.dropout(self.h1, self.keep_prob)
            self.y = tf.matmul(self.h_fc1_drop, self.W2) + self.b2  # output layer

            self.var_list = [self.W1, self.b1, self.W2, self.b2]

            regularizer = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)
            # vanilla single-task loss
            self.cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y)) #+ BETA*regularizer

            # performance metrics
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            self.train_second = optimizer.minimize(self.cross_entropy, var_list=[self.W2, self.b2])
            self.sess.run(tf.global_variables_initializer())

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
                self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_: batch.labels,
                                                          self.keep_prob: 0.5})
                if iter % 1000 == 0:
                    accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                                   feed_dict={
                                                       self.x: test.images, self.y_: test.labels, self.keep_prob: 1.0
                                                   })
                    #print "Iters = %d Accuracy = %f Loss = %f" % (iter, accuracy, loss)
            accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                           feed_dict={self.x: test.images, self.y_: test.labels, self.keep_prob: 1.0})

            # print "Iters = %d Accuracy = %f Loss = %f" % (iters, accuracy, loss)
        return accuracy, loss

    def train_second_layer(self, train, test, iters=1000):
        with self.g.as_default():
            for iter in range(iters):
                batch = train.next_batch(200)
                self.sess.run(self.train_second, feed_dict={
                    self.x: batch.images, self.y_: batch.labels,
                    self.keep_prob: 0.5
                })
                if iter % 100 == 0:
                    accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                                   feed_dict={
                                                       self.x: test.images, self.y_: test.labels, self.keep_prob: 1.0
                                                       })
                    #print "Iters = %d Accuracy = %f Loss = %f" % (iter, accuracy, loss)
            accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                           feed_dict={self.x: test.images, self.y_: test.labels, self.keep_prob: 1.0})

            #print "Iters = %d Accuracy = %f Loss = %f" % (iters, accuracy, loss)
        return accuracy, loss


    def evaluate(self, data):
        """
        :param data: data to evaluate
        :return: (accuracy, loss)
        """
        with self.g.as_default():
            return self.sess.run([self.accuracy, self.cross_entropy],
                                 feed_dict={self.x: data.images, self.y_: data.labels, self.keep_prob: 1.0})


    def compute_fisher(self, images, num_samples=200):
        """
        Calculates fisher information for each variable
        :param images:
        :param num_samples:
        :return:
        """
        with self.g.as_default():
            self.global_count = 0
            # initialize Fisher information for most recent task
            self.F_accum = []
            for v in range(len(self.var_list)):
                self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

            # sampling a random class from softmax
            self.probs = tf.nn.softmax(self.y)
            self.class_ind = tf.to_int32(tf.multinomial(tf.log(self.probs), 1)[0][0])

            for i in range(num_samples):
                im_ind = np.random.randint(images.shape[0])
                # compute first-order derivatives
                ders = self.sess.run(tf.gradients(tf.log(self.probs[0, self.class_ind]), self.var_list),
                                     feed_dict={self.x: images[im_ind:im_ind + 1], self.keep_prob: 1.0})
                # square the derivatives and add to total
                for v in range(len(self.F_accum)):
                    self.F_accum[v] += np.square(ders[v])

            # divide totals by number of samples
            for v in range(len(self.F_accum)):
                self.F_accum[v] /= num_samples


    def get_distance(self, w1_1, b1_1, w2_1, b2_1, w1_2, b1_2, w2_2, b2_2):
        p1 = np.concatenate((w1_1.flatten(), b1_1.flatten(), w2_1.flatten(), b2_1.flatten()))
        p2 = np.concatenate((w1_2.flatten(), b1_2.flatten(), w2_2.flatten(), b2_2.flatten()))
        return np.linalg.norm(p1 - p2)

    def copy(self):
        with self.g.as_default():
            new_m = FlexibleModel(self.num_hidden)
            new_m.set_vars(w1=self.get_w1(), b1=self.get_b1(), w2=self.get_w2(), b2=self.get_b2())
            return new_m

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

    def get_hidden_neuron_space(self, n_index, data, epsilon=0.4, delta=0.9,):
        '''
        We are going to calculate the sphere around a neuron in the hidden layer where with delta probability,
        the loss is less than
        epsilon.
        '''
        #TODO: Check these are getting copied correctly
        #lm = LossModel(self.num_hidden)
        lm = LossCalculator(architecture=[self.num_hidden, 10])
        sample_size = 100
        new_w1 = self.get_w1()
        neuron_w = new_w1[:,n_index]
        new_b1 = self.get_b1()
        neuron_b = new_b1[n_index]
        new_w2 = self.get_w2()
        out_w = new_w2[n_index, :]
        b2 = self.get_b2()
        radius = 1.0
        while True:
            succeed = 0
            for iter in range(sample_size):
                # Handle weights
                w1_pert = np.random.normal(loc=0.0, scale=1.0, size=neuron_w.shape)
                b1_pert = np.random.normal(loc=0.0, scale=1.0, size=neuron_b.shape)
                w2_pert = np.random.normal(loc=0.0, scale=1.0, size=out_w.shape)
                normalizer = 1.0 / np.sqrt(np.sum(np.square(w1_pert)) +
                                           np.sum(np.square(b1_pert)) +
                                           np.sum(np.square(w2_pert))
                                           )
                new_neuron_w = neuron_w + w1_pert * radius * normalizer
                new_neuron_b = neuron_b + b1_pert * radius * normalizer
                new_out_w = out_w + w2_pert * radius * normalizer
                new_w1[:,n_index] = new_neuron_w
                new_b1[n_index] = new_neuron_b
                new_w2[n_index, :] = new_out_w
                loss = lm.get_loss(new_w1, new_b1, new_w2, b2, data)
                #self.logger.info("Iter-%d, Loss = %f" % (iter, loss))
                if loss < epsilon:
                    succeed += 1.0
            prop = succeed / sample_size
            #self.logger.info("Neuron-%d Radius=%d Prop = %f" % (n_index, radius, prop))
            if prop >= delta:
                radius += 1.0
            else:
                #self.logger.info("Neuron-%d Final Radius=%d" % (n_index, radius - 1))
                radius = radius - 1.0
                break
        return radius

    def get_output_spheres(self, n_index, data, delta=0.9, epsilon=0.4):
        '''
                We are going to calculate the sphere around a neuron in the hidden layer where with delta probability,
                the loss is less than
                epsilon.
                '''
        #lm = LossModel(self.num_hidden)
        lm = LossCalculator(architecture=[self.num_hidden, 10])
        sample_size = 20
        new_w2 = self.get_w2()
        neuron_w = new_w2[:, n_index]
        new_b2 = self.get_b2()
        neuron_b = new_b2[n_index]
        w1 = self.get_w1()
        b1 = self.get_b1()
        radius = 0.0
        while True:
            succeed = 0
            for iter in range(sample_size):
                # Handle weights
                w_pert = np.random.normal(loc=0.0, scale=1.0, size=neuron_w.shape)
                b_pert = np.random.normal(loc=0.0, scale=1.0, size=neuron_b.shape)
                normalizer = 1.0 / np.sqrt(np.sum(np.square(w_pert)) +
                                           np.sum(np.square(b_pert)))
                new_neuron_w = neuron_w + w_pert * radius * normalizer
                new_neuron_b = neuron_b + b_pert * radius * normalizer
                new_w2[:, n_index] = new_neuron_w
                new_b2[n_index] = new_neuron_b
                loss = lm.get_loss(w1=w1, b1=b1,w2=new_w2, b2=new_b2, data=data)

                if loss < epsilon:
                    succeed += 1.0
            prop = succeed / sample_size
            #self.logger.info("N-%d R = %f, Prop = %f" % (n_index, radius, prop))
            if prop > delta:
                radius += 0.1
            else:
                # self.logger.info("Neuron-%d Final Radius=%d" % (n_index, radius - 1))
                return radius - 0.1

    def get_imp_output_spheres(self, n_index, data, delta=0.9, epsilon=0.4):
        '''
                We are going to calculate the sphere around a neuron in the hidden layer where with delta probability,
                the loss is less than
                epsilon.
                '''
        #lm = LossModel(self.num_hidden)
        lm = LossCalculator(architecture=[self.num_hidden, 10])
        sample_size = 20
        new_w2 = self.get_w2()
        neuron_w = new_w2[:, n_index]
        new_b2 = self.get_b2()
        neuron_b = new_b2[n_index]
        w1 = self.get_w1()
        b1 = self.get_b1()
        radius = 0.0

        imp_mask = np.zeros(self.num_hidden)
        imp_mask[np.where(neuron_w > np.percentile(neuron_w, 50))] = 1.0

        while True:
            succeed = 0
            for iter in range(sample_size):
                # Handle weights
                w_pert = imp_mask*np.random.normal(loc=0.0, scale=1.0, size=neuron_w.shape)
                #b_pert = np.random.normal(loc=0.0, scale=1.0, size=neuron_b.shape)
                normalizer = 1.0 / np.sqrt(np.sum(np.square(w_pert)))
                new_neuron_w = neuron_w + w_pert * radius * normalizer
                #new_neuron_b = neuron_b + b_pert * radius * normalizer
                new_w2[:, n_index] = new_neuron_w
                #new_b2[n_index] = new_neuron_b
                loss = lm.get_loss(w1=w1, b1=b1,w2=new_w2, b2=new_b2, data=data)

                if loss < epsilon:
                    succeed += 1.0
            prop = succeed / sample_size
            #self.logger.info("N-%d R = %f, Prop = %f" % (n_index, radius, prop))
            if prop > delta:
                radius += 0.1
            else:
                # self.logger.info("Neuron-%d Final Radius=%d" % (n_index, radius - 1))
                return radius - 0.1, imp_mask



class LossModelDEPRECATED:

    def __init__(self, num_hidden):
        self.g = tf.Graph()
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
            self.W1 = tf.placeholder(tf.float32, [self.in_dim, self.num_hidden])
            self.b1 = tf.placeholder(tf.float32, [self.num_hidden])

            self.W2 = tf.placeholder(tf.float32, [self.num_hidden, self.out_dim])
            self.b2 = tf.placeholder(tf.float32, [self.out_dim])

            self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
            self.y = tf.matmul(self.h1, self.W2) + self.b2  # output layer

            self.var_list = [self.W1, self.b1, self.W2, self.b2]

            # vanilla single-task loss
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
            self.sess.run(tf.global_variables_initializer())

    def get_loss(self, w1, b1, w2, b2, data):
        fd = {self.x: data.images, self.y_: data.labels, self.W1: w1, self.b1: b1, self.W2: w2, self.b2: b2}
        with self.g.as_default():
            return self.sess.run(self.cross_entropy,feed_dict=fd)


