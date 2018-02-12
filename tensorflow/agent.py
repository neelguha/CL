import tensorflow as tf
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
from model_types.mnist_cnn import *
from model_types.mnist_logit import *
from model_types.two_layer_nn import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Agent:

    def __init__(self, agent_id, regularize = False):
        self.agent_id = agent_id
        self.train, self.test = get_data(self.agent_id)
        self.model = MnistLogit(regularize)
        self.opt = None
        self.opt_loss = None
        self.opt_accuracy = None

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def get_loss(self, weight, bias):
        self.update_weights(weight, bias)
        return self.model.cost(self.train)

    def get_accuracy(self, weight, bias):
        self.update_weights(weight, bias)
        return self.model.get_accuracy(self.train)

    def update_weights(self, w_obj):
        self.model.update_w(w_obj.w)
        self.model.update_b(w_obj.b)

    def get_loss_accuracy(self, w_object, use_test = False):
        self.update_weights(w_object)
        if use_test:
            return self.model.get_accuracy_cost(self.test)
        return self.model.get_accuracy_cost(self.train)

    def get_optimum(self, good= True):
        if self.opt == None:
            self.model.train_model(self.train, good)
            self.opt_loss, self.opt_accuracy= self.model.get_accuracy_cost(self.test)
        return self.opt, self.opt_loss, self.opt_accuracy

    def get_gradient(self):
         dw, db = self.model.get_gradients(self.train)
         return Weight(w=dw, b=db)

    def get_hessian(self):
        dw, db = self.model.get_hessian(self.train)
        return Weight(w=dw, b=db)






