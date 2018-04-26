"""
Useful functions for saving/loading models from file.
"""

import os
import numpy as np
import tensorflow as tf

################## MODEL IO OPERATIONS #####################

def get_file_identifier(architecture):
    """ Converts architecture into model identifier for files"""
    return '_'.join([str(x) for x in architecture])

def save_model(model, model_name):
    identifier = get_file_identifier(model.architecture)
    output_dir = "mt_models/%s_%s" % (model_name, identifier)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    w1, b1 = model.get_parameters(0)
    w2, b2 = model.get_parameters(1)

    np.save("%s/w1" % output_dir, w1)
    np.save("%s/b1" % output_dir, b1)
    np.save("%s/w2" % output_dir, w2)
    np.save("%s/b2" % output_dir, b2)


def load_model(model_name, num_hidden, model_type):
    model_dir = "mt_models/%s_%d" % (model_name, num_hidden)
    w1 = np.load("%s/w1.npy" % model_dir)
    b1 = np.load("%s/b1.npy" % model_dir)
    w2 = np.load("%s/w2.npy" % model_dir)
    b2 = np.load("%s/b2.npy" % model_dir)
    new_model = model_type(num_hidden)
    new_model.set_vars(w1, b1, w2, b2)
    return new_model

def load_model_with_arch(model_name, architecture, model_type):
    """ Load model with passed architecture object. Necessary for layer-wise combination. """
    identifier = get_file_identifier(architecture)
    identifier = "50"
    model_dir = "mt_models/%s_%s" % (model_name, identifier)
    w1 = np.load("%s/w1.npy" % model_dir)
    b1 = np.load("%s/b1.npy" % model_dir)
    w2 = np.load("%s/w2.npy" % model_dir)
    b2 = np.load("%s/b2.npy" % model_dir)
    new_model = model_type(architecture)
    new_model.set_vars([(w1, b1), (w2, b2)])
    return new_model

################# MODEL OPERATIONS #########################

def average_models(m1, m2):
    new_m = m1.copy()

    m1w1, m1b1 = m1.get_parameters(0)
    m1w2, m1b2 = m1.get_parameters(1)

    m2w1, m2b1 = m2.get_parameters(0)
    m2w2, m2b2 = m2.get_parameters(1)


    new_w1 = (m1w1 + m2w1) / 2.0
    new_b1 = (m1b1 + m2b1) / 2.0
    new_w2 = (m1w2 + m2w2) / 2.0
    new_b2 = (m1b2 + m2b2) / 2.0
    new_m.set_vars([(new_w1, new_b1), (new_w2, new_b2)])
    return new_m

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


################### NEURON OPERATIONS #######################


def average_hidden_neuron_radius_wise(m1, m2, n1, n2, r1, r2):
    d = get_full_neuron_distance(n1, n2, m1, m2)
    r1 = min(d, r1)
    r2 = min(d, r2)

    m1w1, m1b1 = m1.get_parameters(0)
    m1w2, _ = m1.get_parameters(1)

    m2w1, m2b1 = m2.get_parameters(0)
    m2w2, _ = m2.get_parameters(1)

    m1_nw_in = m1w1[:, n1]

    m1_nw_out = m1w2[n1, :]

    m1_nb = m1b1[n1]

    m2_nw_in = m2w1[:, n2]

    m2_nw_out = m2w2[n2, :]

    m2_nb = m2b1[n2]


    r1_bound_w_in = (r1 / d) * (m2_nw_in - m1_nw_in) + m1_nw_in
    r2_bound_w_in = ((d - r2) / d) * (m2_nw_in - m1_nw_in) + m1_nw_in

    r1_bound_w_out = (r1 / d) * (m2_nw_out - m1_nw_out) + m1_nw_out
    r2_bound_w_out = ((d - r2) / d) * (m2_nw_out - m1_nw_out) + m1_nw_out

    r1_bound_b = (r1 / d) * (m2_nb - m1_nb) + m1_nb
    r2_bound_b = ((d - r2) / d) * (m2_nb - m1_nb) + m1_nb

    new_w_in = (r1_bound_w_in + r2_bound_w_in) / 2.0
    new_b = (r1_bound_b + r2_bound_b) / 2.0
    new_w_out = (r1_bound_w_out + r2_bound_w_out) / 2.0
    return new_w_in, new_b, new_w_out

def get_neuron_distance(layer, n1, n2, m1, m2):
    """
    Calculate distance between two neurons in the same layer of different models using
    the weights and biases of each neuron as coordinates
    """
    if layer == 1:
        m1w = m1.get_w1()[:, n1]
        m2w = m2.get_w1()[:, n2]
        m1b = m1.get_b1()[n1]
        m2b = m2.get_b1()[n2]
    elif layer == 2:
        m1w = m1.get_w2()[:, n1]
        m2w = m2.get_w2()[:, n2]
        m1b = m1.get_b2()[n1]
        m2b = m2.get_b2()[n2]
    else:
        raise Exception("Error calculating distance.")

    dist = np.sum(np.square(m1w - m2w)) + \
           np.sum(np.square(m1b - m2b))
    dist = np.sqrt(dist)
    return dist

def get_full_neuron_distance(n1, n2, m1, m2):
    """
    Calculate the distance between two neurons in two models using incoming weights, neuron bias, and outgoing
    bias.
    """
    m1w1, m1b1 = m1.get_parameters(0)
    m1w2, _ = m1.get_parameters(1)

    m2w1, m2b1 = m2.get_parameters(0)
    m2w2, _ = m2.get_parameters(1)

    dist = np.sum(np.square(m1w1[:, n1] - m2w1[:, n2])) + \
        np.sum(np.square(m1b1[n1] - m2b1[n2])) + \
        np.sum(np.square(m1w2[n1, :] - m2w2[n2, :]))
    dist = np.sqrt(dist)
    return dist

'''

def average_hidden_neuron_radius_wise(m1, m2, n1, n2, r1, r2):
    d = get_full_neuron_distance(n1, n2, m1, m2)
    r1 = min(d, r1)
    r2 = min(d, r2)

    m1_w1 = m1.get_w1()
    m1_nw_in = m1_w1[:, n1]

    m1_w2 = m1.get_w2()
    m1_nw_out = m1_w2[n1, :]

    m1_b = m1.get_b1()
    m1_nb = m1_b[n1]

    m2_w1 = m2.get_w1()
    m2_nw_in = m2_w1[:, n2]

    m2_w2 = m2.get_w2()
    m2_nw_out = m2_w2[n2, :]

    m2_b = m2.get_b1()
    m2_nb = m2_b[n2]


    r1_bound_w_in = (r1 / d) * (m2_nw_in - m1_nw_in) + m1_nw_in
    r2_bound_w_in = ((d - r2) / d) * (m2_nw_in - m1_nw_in) + m1_nw_in

    r1_bound_w_out = (r1 / d) * (m2_nw_out - m1_nw_out) + m1_nw_out
    r2_bound_w_out = ((d - r2) / d) * (m2_nw_out - m1_nw_out) + m1_nw_out

    r1_bound_b = (r1 / d) * (m2_nb - m1_nb) + m1_nb
    r2_bound_b = ((d - r2) / d) * (m2_nb - m1_nb) + m1_nb

    new_w_in = (r1_bound_w_in + r2_bound_w_in) / 2.0
    new_b = (r1_bound_b + r2_bound_b) / 2.0
    new_w_out = (r1_bound_w_out + r2_bound_w_out) / 2.0
    return new_w_in, new_b, new_w_out
'''