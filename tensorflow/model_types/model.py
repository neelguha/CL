import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import os

'''
This is the basic class for every type of model we construct

'''
class ModelTemplate:

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.train, self.test = self.get_data(agent_id)
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.var_list = []
        self.var_counts = []
        self.flat_vars = []
        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, 10])

    def fill_model(self):
        self.parameter_count = self.get_nb_params_shape()
        self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        for var in self.var_list:
            self.flat_vars.append(var.fla)


    '''
    Trains the model on the provided train data. 
    '''
    def train_model(self, iters = 2000):
        for _ in range(iters):
            batch = self.train.next_batch(200)
            self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_:   batch.labels})

    '''
    Returns (loss, accuracy) of model on particular dataset
    '''
    def get_accuracy_cost(self, data = None):
        if not data == None:
            return self.sess.run([self.loss, self.accuracy], feed_dict={self.x: data.images,
                                                       self.y_: data.labels})
        return self.sess.run([self.loss, self.accuracy], feed_dict={self.x: self.test.images,
                                                                    self.y_: self.test.labels})

    '''
    Get num vars
    '''
    def count_number_trainable_params(self):
        '''
        Counts the number of trainable variables.
        '''
        tot_nb_params = 0
        for trainable_variable in self.var_list:
            shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
            current_nb_params = self.get_nb_params_shape(shape)
            self.var_counts.append(current_nb_params)
            tot_nb_params = tot_nb_params + current_nb_params
        return tot_nb_params

    def get_nb_params_shape(self, shape):
        '''
        Computes the total number of params for a given shap.
        Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
        '''
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    def get_param(self, index):

        for i,var in enumerate(self.var_list):
            if index > self.var_counts[i]:

    def get_gradients(self, data):
        with self.sess.as_default():
            dc_dw, dc_db = tf.gradients(self.loss, [self.W, self.b], )
        a,b = self.sess.run([dc_dw, dc_db], feed_dict={self.x: data.images,
                                                       self.y_: data.labels})
        return a,b



