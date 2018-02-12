import tensorflow as tf
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from agent import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class JointOptimizer():

    def __init__(self, a0, a1):
        # We're trying to solve the following optimization problem
        # minimize {(w - W1)G1 + (w - W2)G2) subject to w
        W0, _, _ = a0.get_optimum()
        W1, _, _ = a1.get_optimum()
        W0 = W0.joint()
        W1 = W1.joint()
        G0 = a0.get_gradient().joint()
        G1 = a1.get_gradient().joint()
        self.sess = tf.Session()
        initial_value = np.array(W0) + np.array(W1) / 2
        w = tf.Variable(tf.zeros(785*10))
        self.loss = tf.reduce_sum(tf.abs(tf.multiply(tf.square(tf.subtract(w,W0)), G0)) + tf.abs(tf.multiply(tf.square(tf.subtract(w,W1)), G1)))
        self.train_step = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for i in range(3000):
            self.sess.run(self.train_step)

        w_val = self.sess.run(w)
        w_obj = Weight(joined=w_val)
        l = self.sess.run(self.loss)
        print "Iter %d Loss: %f" % (i, l)
        cross_test(a0, a1, w_obj, w_obj)


def cross_test(a0, a1, a0_opt, a1_opt):
    print "Cross Testing:"
    a0_cost, a0_acc = a0.get_loss_accuracy(a1_opt, use_test=True)
    a1_cost, a1_acc = a1.get_loss_accuracy(a0_opt, use_test=True)
    print "New Agent 0: Cost: %f Accuracy: %f" % (a0_cost, a0_acc)
    print "New Agent 1: Cost: %f Accuracy: %f" % (a1_cost, a1_acc)

def main():
    a0 = Agent(0)
    a1 = Agent(1)
    a0_opt, a0_opt_loss, a0_opt_accuracy = a0.get_optimum()
    print "Agent 0. Optimal Cost: %f Optimal Accuracy: %f" % (a0_opt_loss, a0_opt_accuracy)
    a0_grad = a0.get_gradient()
    a1_opt, a1_opt_loss, a1_opt_accuracy  = a1.get_optimum()
    print "Agent 1. Optimal Cost: %f Optimal Accuracy: %f" % (a1_opt_loss, a1_opt_accuracy)
    a1_grad = a1.get_gradient()

    cross_test(a0, a1, a0_opt, a1_opt)
    JointOptimizer(a0, a1)

    '''for x,y,z,d in zip(a0_opt.joint(), a1_opt.joint(), a0_grad.joint(), a1_grad.joint()):
        if x + y + z + d == 0:
            continue
        print "A0_opt: %f, A1_opt: %f, A0_g: %f, A1_g: %f" % (x,y,z,d)'''

if __name__ == '__main__':
    main()