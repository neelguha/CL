"""
Experiment Version 4:
Given two models, first realign the models, then perform fisher based mapping to combine the models

"""
from mt_model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging, coloredlogs
from controller import *
from realignment import *
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

# Constants
AGENT1 = [0, 1, 2, 3, 4]
AGENT2 = [5, 6, 7, 8, 9]
NUM_OUTPUT = 10
# Logging initialization
logger = logging.getLogger(__name__)
formatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Command line flags initialization
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("hidden", 50, "Number of units in hidden layer 1.")
flags.DEFINE_bool("new", False, "Whether to build new models")

def average_models(m1, m2):
    new_m = MTModel(m1.num_hidden, logger)
    avg_w1 = (m1.get_w1() + m2.get_w1()) / 2.0
    avg_b1 = (m1.get_b1() + m2.get_b1()) / 2.0
    avg_w2 = (m1.get_w2() + m2.get_w2()) / 2.0
    avg_b2 = (m1.get_b2() + m2.get_b2()) / 2.0
    new_m.set_vars(avg_w1, avg_b1, avg_w2, avg_b2)
    return new_m

def save_model(model, model_name):
    output_dir = "mt_models/%s_%d" % (model_name, model.num_hidden)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save("%s/w1" % output_dir, model.get_w1())
    np.save("%s/b1" % output_dir, model.get_b1())
    np.save("%s/w2" % output_dir, model.get_w2())
    np.save("%s/b2" % output_dir, model.get_b2())

def load_model(model_name, num_hidden):
    model_dir = "mt_models/%s_%d" % (model_name, num_hidden)
    w1 = np.load("%s/w1.npy" % model_dir)
    b1 = np.load("%s/b1.npy" % model_dir)
    w2 = np.load("%s/w2.npy" % model_dir)
    b2 = np.load("%s/b2.npy" % model_dir)
    new_model = MTModel(num_hidden, logger)
    new_model.set_vars(w1, b1, w2, b2)
    return new_model


def run_tests(mt, digits1, digits2, gold):
    # TODO: Print these out as table
    logger.info("Evaluating models on all data...")
    accuracy, loss = gold.evaluate(mt.all_test)
    logger.info("GOLD Loss= %f Accuracy = %f" % (loss, accuracy))
    accuracy, loss = digits1.evaluate(mt.all_test)
    logger.info("DIGIT1 Loss= %f Accuracy = %f" % (loss, accuracy))
    accuracy, loss = digits2.evaluate(mt.all_test)
    logger.info("DIGIT2: Loss= %f Accuracy = %f" % (loss, accuracy))
    # Average models
    avg_m = average_models(digits1, digits2)
    accuracy, loss = avg_m.evaluate(mt.all_test)
    logger.info("AVERAGED: Loss= %f Accuracy = %f" % (loss, accuracy))

def new_models(mt):
    logger.info("Creating new models... Num hidden = %d" % FLAGS.hidden)
    w1 = np.random.normal(0.0, 0.1, (784, FLAGS.hidden))
    b1 = np.random.normal(0.0, 0.1, (FLAGS.hidden))
    w2 = np.random.normal(0.0, 0.1, (FLAGS.hidden, NUM_OUTPUT))
    b2 = np.random.normal(0.0, 0.1, (NUM_OUTPUT))
    gold = MTModel(FLAGS.hidden, logger)
    gold.set_vars(w1, b1, w2, b2)
    logger.info("Training gold...")
    gold.train_model(mt.all_train, mt.all_test)
    save_model(gold, "gold")

    digits1 = MTModel(FLAGS.hidden, logger)
    digits1.set_vars(w1, b1, w2, b2)
    logger.info("Training mnist1 digits...")
    digits1.train_model(mt.mnist1_train, mt.mnist1_test)
    save_model(digits1, "digit1")

    digits2 = MTModel(FLAGS.hidden, logger)
    digits2.set_vars(w1, b1, w2, b2)
    logger.info("Training mnist2 digits...")
    digits2.train_model(mt.mnist2_train, mt.mnist2_test)
    save_model(digits2, "digit2")
    return digits1, digits2, gold

def load_models():
    logger.info("Loading model from mt_models/digit1_%d" % FLAGS.hidden)
    digits1 = load_model("digit1", FLAGS.hidden)
    logger.info("Loading model from mt_models/digit2_%d" % FLAGS.hidden)
    digits2 = load_model("digit2", FLAGS.hidden)
    logger.info("Loading model from mt_models/gold_%d" % FLAGS.hidden)
    gold = load_model("gold", FLAGS.hidden)
    return digits1, digits2, gold

def get_hidden_distance(n1, n2, m1, m2):
    m1n = np.append(m1.get_w1()[:, n1], m1.get_b1()[n1])
    m2n = np.append(m2.get_w1()[:, n2], m2.get_b1()[n2])
    return np.linalg.norm(m1n - m2n)

def get_output_distance(n1, n2, m1, m2):
    m1n = np.append(m1.get_w2()[:, n1], m1.get_b2()[n1])
    m2n = np.append(m2.get_w2()[:, n2], m2.get_b2()[n2])
    return np.linalg.norm(m1n - m2n)

def get_output_mask_distance(n1, n2, mask1, mask2, m1, m2):
    m1n = m1.get_w2()[:, n1]*mask1
    m2n = m2.get_w2()[:, n2]*mask2
    dist = 0.0
    for i in range(FLAGS.hidden):
        if m1n[i] == 0.0 or m2n[i] == 0.0:
            continue
        dist = dist + np.square(m1n[i] - m2n[i])
    return np.sqrt(dist)
    #return np.linalg.norm(m1n - m2n)


def average_hidden_neuron_radius_wise(m1, m2, n1, n2, r1, r2):
    d = get_hidden_distance(n1, n2, m1, m2)
    r1 = min(d, r1)
    r2 = min(d, r2)
    m1_w = m1.get_w1()
    m1_nw = m1_w[:, n1]

    m1_b = m1.get_b1()
    m1_nb = m1_b[n1]

    m2_w = m2.get_w1()
    m2_nw = m2_w[:, n2]

    m2_b = m2.get_b1()
    m2_nb = m2_b[n2]


    r1_bound_w = (r1 / d) * (m2_nw - m1_nw) + m1_nw
    r2_bound_w = ((d - r2) / d) * (m2_nw - m1_nw) + m1_nw

    r1_bound_b = (r1 / d) * (m2_nb - m1_nb) + m1_nb
    r2_bound_b = ((d - r2) / d) * (m2_nb - m1_nb) + m1_nb

    m1_w[:, n1] = (r1_bound_w + r2_bound_w) / 2.0
    m1_b[n1] = (r1_bound_b + r2_bound_b) / 2.0
    m1.set_vars(w1=m1_w, b1=m1_b)


def average_output_neuron_radius_wise(m1, m2, n1, n2, d1_mask, d2_mask, r1, r2):
    d = get_hidden_distance(n1, n2, m1, m2)
    r1 = min(d, r1)
    r2 = min(d, r2)
    m1_w = m1.get_w2()
    m1_nw = m1_w[:, n1]

    m1_b = m1.get_b2()
    m1_nb = m1_b[n1]

    m2_w = m2.get_w2()
    m2_nw = m2_w[:, n2]

    m2_b = m2.get_b2()
    m2_nb = m2_b[n2]

    for i in range(FLAGS.hidden):
        if d1_mask[i] == 0 and d2_mask[i] == 0:
            m1_w[i, n1] = (m2_nw[i] + m1_nw[i]) / 2.0
        elif d1_mask[i] == 0 and d2_mask[i] != 0:
            m1_w[i, n1] = m2_nw[i]
        elif d1_mask[i] != 0 and d2_mask[i] == 0:
            m1_w[i, n1] = m1_nw[i]
        elif d1_mask[i] != 0 and d2_mask[i] != 0:
            r1_bound = (r1 / d) * (m2_nw[i] - m1_nw[i]) + m1_nw[i]
            r2_bound = ((d - r2) / d) * (m2_nw[i] - m1_nw[i]) + m1_nw[i]
            m1_w[i, n1] = (r1_bound + r2_bound) / 2.0
        else:
            print "ERROR"

    m1.set_vars(w2=m1_w, b2=m1_b)


def main():

    mt = MTData()
    logger.info("Loaded all data..")
    if FLAGS.new:
        digits1, digits2, gold = new_models(mt)
    else:
        digits1, digits2, gold = load_models()

    run_tests(mt, digits1,digits2,gold)
    sample = mt.mnist1_train.next_batch(50)
    delta = 0.7
    epsilon = 1.0
    logger.info("Delta=%f\tEpsilon=%f" % (delta, epsilon))
    logger.info("Realigning hidden neurons...")
    d1_radii = []
    for i in range(FLAGS.hidden):
        r = digits1.get_hidden_sphere(i,sample, delta=delta, epsilon=epsilon)
        d1_radii.append(r)

    sample = mt.mnist2_train.next_batch(50)
    d2_radii = []
    for i in range(FLAGS.hidden):
        r = digits2.get_hidden_sphere(i, sample, delta=delta, epsilon=epsilon)
        d2_radii.append(r)


    costs = np.ones((FLAGS.hidden, FLAGS.hidden))
    for d1_n in range(FLAGS.hidden):
        for d2_n in range(FLAGS.hidden):
            dist = get_hidden_distance(d1_n, d2_n, digits1, digits2)
            if dist < d1_radii[d1_n] + d2_radii[d2_n]:
                #logger.info("N-%d (R=%f) and N-%d (R=%f), d = %f" % (d1_n, d1_radii[d1_n], d2_n, d2_radii[d2_n], dist))
                costs[d1_n, d2_n] = 0.0

    d1_neurons, d2_neurons =  linear_sum_assignment(costs)
    combined = digits1.copy()
    for d1n, d2n, d1r, d2r in zip(d1_neurons, d2_neurons, d1_radii, d2_radii):
        #logger.info("Pairing N-%d with N-%d" % (d1n, d2n))
        average_hidden_neuron_radius_wise(combined, digits2, d1n, d2n, d1r, d2r)
        #average_neuron(combined, digits2, d1n, d2n)

    '''accuracy, loss = digits1.evaluate(mt.mnist1_test)
    logger.info("UNALIGNED DIGIT1 on MNIST1 Loss= %f Accuracy = %f" % (loss, accuracy))
    accuracy, loss = combined.evaluate(mt.mnist1_test)
    logger.info("REALIGNED DIGIT1 on MNIST1 Loss= %f Accuracy = %f" % (loss, accuracy))'''


    logger.info("perturbing output neurons")
    sample = mt.mnist1_train.next_batch(50)
    d1_radii = []
    d1_masks = []
    for i in range(10):
        #r = digits1.get_output_spheres(i, sample, delta=delta, epsilon=epsilon)
        r, mask = digits1.get_imp_output_spheres(i, sample, delta, epsilon)
        d1_masks.append(mask)
        d1_radii.append(r)

    sample = mt.mnist2_train.next_batch(50)
    d2_radii = []
    d2_masks = []
    for i in range(10):
        #r = digits2.get_output_spheres(i, sample, delta=delta, epsilon=epsilon)
        r, mask = digits2.get_imp_output_spheres(i, sample, delta, epsilon)
        d2_masks.append(mask)
        d2_radii.append(r)

    for i in range(10):
        dist = get_output_mask_distance(i, i, d1_masks[i], d2_masks[i], digits1, digits2)
        if dist < d1_radii[i] + d2_radii[i]:
            logger.info("SUCCESS. N-%d. R1 = %f, R2 = %f, D = %f" % (i, d1_radii[i], d2_radii[i], dist))
        else:
            logger.info("FAILURE. N-%d. R1 = %f, R2 = %f, D = %f" % (i, d1_radii[i], d2_radii[i], dist))
        average_output_neuron_radius_wise(combined, digits2, i, i, d1_masks[i], d2_masks[i], d1_radii[i], d2_radii[i])
        '''print np.round(abs((digits1.get_w2()[:, i])).tolist(), 2)
        print np.round(abs((digits1.get_b2()[i])).tolist(), 2)
        print np.round(abs((digits2.get_w2()[:, i])).tolist(), 2)
        print np.round(abs((digits2.get_b2()[i])).tolist(), 2)'''

    accuracy, loss = combined.evaluate(mt.all_test)
    logger.info("COMBINED: Loss= %f Accuracy = %f" % (loss, accuracy))



    '''
    m1_cubes = []
    m2_cube = []
    for n in m1.hidden:
        m1_cubes.append(hypercube(n))
    // repeat
    for m2
        matches, non_matches = align(m1_cubes, m2_cubes)
    // Create new network with matched neurons
    // Add unmatched neurons into second network
    '''


if __name__ == '__main__':
    main()