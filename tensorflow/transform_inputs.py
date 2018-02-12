import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from model_types import mnist_logit
from data import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)



NUM_NEURONS = 2 # The input size for the second layer



mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train = Data(mnist.train.images, mnist.train.labels)
test = Data(mnist.test.images, mnist.test.labels)

m = mnist_logit.MnistLogit(num_neurons = NUM_NEURONS)
m.train_model(train, test, iters=4000)
train_activations =  m.l1_activations(train)
test_activations = m.l1_activations(test)


# Write full data to file
np.savetxt("transformed_inputs/size_%d/train_acts.txt" % NUM_NEURONS, train_activations)
np.savetxt("transformed_inputs/size_%d/train_labels.txt" % NUM_NEURONS, mnist.train.labels)
np.savetxt("transformed_inputs/size_%d/test_acts.txt"% NUM_NEURONS, test_activations)
np.savetxt("transformed_inputs/size_%d/test_labels.txt" % NUM_NEURONS, mnist.test.labels)



# Split by digits and write to files
for digit in range(10):
    digit_labels = []
    digit_vals = []
    print "Writing train for D-%d" % digit
    for i,train_label in enumerate(mnist.train.labels):
        if train_label[digit] == 1.0:
            digit_labels.append(train_label)
            digit_vals.append(train_activations[i])
    np.savetxt("transformed_inputs/size_%d/digit_%d_train_acts.txt" % (NUM_NEURONS, digit), digit_vals)
    np.savetxt("transformed_inputs/size_%d/digit_%d_train_labels.txt" % (NUM_NEURONS, digit), digit_labels)

    print "Writing test for D-%d" % digit
    test_digit_labels = []
    test_digit_vals = []
    for i, test_label in enumerate(mnist.test.labels):
        if test_label[digit] == 1.0:
            test_digit_labels.append(test_label)
            test_digit_vals.append(test_activations[i])
    np.savetxt("transformed_inputs/size_%d/digit_%d_test_acts.txt" % (NUM_NEURONS,digit), test_digit_vals)
    np.savetxt("transformed_inputs/size_%d/digit_%d_test_labels.txt" % (NUM_NEURONS,digit), test_digit_labels)



