"""
Implements the model for layer-wise learning
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from loss_calculator import *
from model_utils import *
import tensorflow as tf
INPUT_SIZE = 784
OUTPUT_SIZE = 10  # 10 digits
BETA = 0.01

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
            w1, b1 = m.get_parameters(0)
            w2, b2 = m.get_parameters(1)

            for i in range(m.num_hidden):
                self.add_hidden_neuron(w1[:, i], w2[i, :], b1[i])
            self.layer2_b = self.layer2_b + b2
        self.layer2_b = self.layer2_b / 2.0
        return self.convert_to_model()


    def convert_to_model(self):
        architecture = [self.hidden, 10]
        model = AgentModel(architecture)
        model.set_vars([(self.layer1_w, self.layer1_b), (self.layer2_w, self.layer2_b)])
        return model


class AgentModel:

    def __init__(self, architecture):
        """
        Initializes model with given architecture
        """
        self.architecture = architecture
        self.g = tf.Graph()

        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.Session()

            self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
            self.y_ = tf.placeholder(tf.float32, [None, architecture[-1]])  # output placeholder (for calculating loss)
            self.y = None
            self.keep_prob = tf.placeholder(tf.float32)

            self.parameters = []
            for layer, num_neurons in enumerate(architecture):
                b = bias_variable([num_neurons])
                if layer == 0:
                    w = weight_variable([784, num_neurons])
                    self.y = tf.matmul(self.x, w) + b
                else:
                    w = weight_variable([architecture[layer - 1], num_neurons])
                    self.y = tf.matmul(self.y, w) + b
                self.parameters.append((w, b))
                # If hidden layer
                #TODO: THIS WON'T WORK WITH MORE THAN 2 LAYERS
                if layer < len(architecture) - 1:
                    self.drop_out = tf.nn.dropout(self.y, self.keep_prob)
                    self.y = tf.nn.relu(self.drop_out)

            # vanilla single-task loss
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(0.01)
            w2, b2 = self.parameters[-1]
            self.train_second = optimizer.minimize(self.cross_entropy, var_list=[w2, b2])
            self.sess.run(tf.global_variables_initializer())

    def train_model(self, train, test, iters=2000):

        with self.g.as_default():
            for iter in range(iters):
                batch = train.next_batch(200)
                fd = {self.x: batch.images, self.y_: batch.labels, self.keep_prob: 0.5}
                self.sess.run(self.train_step, feed_dict= fd)
            accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict= fd)
        return accuracy, loss

    def train_second_layer(self, train, test, iters=1000):
        with self.g.as_default():
            for iter in range(iters):
                batch = train.next_batch(200)
                self.sess.run(self.train_second, feed_dict={
                    self.x: batch.images, self.y_: batch.labels, self.keep_prob: 1.0
                })
                if iter % 100 == 0:
                    accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy],
                                                   feed_dict={
                                                       self.x: test.images, self.y_: test.labels, self.keep_prob:1.0})
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


    def copy(self):
        with self.g.as_default():
            new_m = AgentModel(self.architecture)
            parameters = [self.get_parameters(0), self.get_parameters(1)]
            new_m.set_vars(parameters)
            return new_m

    def set_vars(self, model_parameters):
        with self.g.as_default():
            for i, params in enumerate(model_parameters):
                w, b = params
                if len(w) > 0:
                    self.sess.run(tf.assign(self.parameters[i][0], w))
                if len(b) > 0:
                    self.sess.run(tf.assign(self.parameters[i][1], b))


    def get_parameters(self, layer):
        return self.sess.run(self.parameters[layer])

    def get_hidden_neuron_space(self, n_index, layer, data, epsilon=0.4, delta=0.9,):
        '''
        We are going to calculate the sphere around a neuron in the hidden layer where with delta probability,
        the loss is less than
        epsilon.
        '''
        #TODO: Check these are getting copied correctly
        lm = LossCalculator(self.architecture)
        sample_size = 100

        new_w1, new_b1 = self.get_parameters(0)
        neuron_w = new_w1[:,n_index]
        neuron_b = new_b1[n_index]
        new_w2, b2 = self.get_parameters(1)
        out_w = new_w2[n_index, :]
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
        the loss is less than epsilon.
        '''
        lm = LossCalculator(self.architecture)
        sample_size = 20
        new_w2, new_b2 = self.get_parameters(1)
        neuron_w = new_w2[:, n_index]
        neuron_b = new_b2[n_index]
        w1, b1 = self.get_parameters(0)
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