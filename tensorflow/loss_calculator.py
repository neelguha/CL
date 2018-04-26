"""
Implements the loss calculator. For a given model architecture, produces a corresponding model with weights
as placeholders. This speeds up loss calculation for a given set of weights.
"""

import tensorflow as tf

class LossCalculator:

    #TODO: Add dropout
    def __init__(self, architecture):
        """
        Initializes a loss calculator object

        Parameters
        ----------
        architecture: a list where the index corresponds to the layer and the value
        corresponds to the number of hidden units at that layer. For example, if
        architecture = [30, 10], we would build a loss calculator for a network with 30
        hidden neurons in the first layer, and 10 output neurons. All hidden neurons use RELU and
        all output neurons use softmax.

        """
        self.g = tf.Graph()
        with self.g.as_default():
            self.sess = tf.Session()
            out_dim = architecture[-1]

            self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
            self.y_ = tf.placeholder(tf.float32, [None, architecture[-1]]) # output placeholder (for calculating loss)
            self.y =  None
            self.weights = []
            for layer, num_neurons in enumerate(architecture):
                b = tf.placeholder(tf.float32, [num_neurons])
                if layer == 0:
                    w = tf.placeholder(tf.float32, [784, num_neurons])
                    self.y = tf.matmul(self.x, w) + b
                else:
                    w = tf.placeholder(tf.float32, [architecture[layer-1], num_neurons])
                    self.y = tf.matmul(self.y, w) + b
                self.weights.append((w, b))
                # If hidden layer
                if layer < len(architecture) - 1:
                    self.y = tf.nn.relu(self.y)

            # vanilla single-task loss
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))
            self.sess.run(tf.global_variables_initializer())

    def get_loss(self, w1, b1, w2, b2, data):
        fd = {self.x: data.images,
              self.y_: data.labels,
              self.weights[0][0]: w1,
              self.weights[0][1]: b1,
              self.weights[1][0]: w2,
              self.weights[1][1]: b2}

        with self.g.as_default():
            return self.sess.run(self.cross_entropy,feed_dict=fd)