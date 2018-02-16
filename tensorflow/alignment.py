from tensorflow.examples.tutorials.mnist import input_data
from model_types import mnist_logit
from data import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
from scipy.optimize import linear_sum_assignment
import logging, coloredlogs


logger = logging.getLogger(__name__)
formatter = coloredlogs.ColoredFormatter(fmt='%(asctime)s %(levelname)s %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

HIDDEN_LAYER_SIZE = 20 # The input size for the second layer

parser = argparse.ArgumentParser(description='MNIST Alignment (Tensorflow)')
parser.add_argument("--new_model", default=False, action="store_true" , help="Whether to create a new model or not")
parser.add_argument("--eo", default=False, action="store_true", help="Build models for Even/Odd task")
args = parser.parse_args()

#TODO: Clean this up
def sort_data(agent1, agent2, images, labels):
    """
    even: [1, 0]
    odd:  [0, 1]
    :param agent1: Digits learned by agent 1
    :param agent2: Digits learned by agent 2
    :param images:
    :param labels:
    :return:
    """
    a1_data = []
    a1_labels = []
    a2_data = []
    a2_labels = []

    data = zip(labels, images)
    for label, image in data:
        digit = np.argmax(label)
        if args.eo:
            even_odd = digit % 2
            image_label = np.zeros(2)
            image_label[even_odd] = 1.0
        else:
            image_label = label
        if digit in agent1:
            a1_labels.append(image_label)
            a1_data.append(image)
        elif digit in agent2:
            a2_labels.append(image_label)
            a2_data.append(image)

    return Data(np.array(a1_data), np.array(a1_labels)), Data(np.array(a2_data), np.array(a2_labels))


def init(agent1, agent2):
    """
    :return: (Train data, test data)
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    a1_train, a2_train = sort_data(agent1, agent2, mnist.train.images, mnist.train.labels)
    a1_test, a2_test = sort_data(agent1, agent2, mnist.test.images, mnist.test.labels)
    logger.info("Sorted data.")
    return a1_train, a1_test, a2_train, a2_test

def save_model(model, name):
    """
    Saves the weights in a model to alignment/name/.
    :param model: A model object to save
    :param name: The subdirectory to save the model to
    """
    output_dir = "alignment/%s" % name
    np.save("%s/w1" % output_dir, model.get_w1())
    np.save("%s/w2" % output_dir, model.get_w2())
    np.save("%s/b1" % output_dir, model.get_b1())
    np.save("%s/b2" % output_dir, model.get_b2())


def copy_model(m):
    return mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE,
                           out_dim=m.out_dim,
                           w1=m.get_w1(),
                           w2=m.get_w2(),
                           b1=m.get_b1(),
                           b2=m.get_b2())

def load_model(name):
    """
    Loads weights corresponding to the model name passed
    :param name: The name of the model to fetch
    :return: returns a model initialized with these weights.
    """
    output_dir = "alignment/%s" % name
    w1 = np.load("%s/w1.npy" % output_dir)
    w2 = np.load("%s/w2.npy" % output_dir)
    b1 = np.load("%s/b1.npy" % output_dir)
    b2 = np.load("%s/b2.npy" % output_dir)
    if args.eo:
        return mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE, out_dim=2, w1=w1, w2=w2, b1=b1, b2=b2)
    else:
        return mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE, out_dim=10, w1=w1, w2=w2, b1=b1, b2=b2)

def get_new_random_model(train, test):
    """
    :param train: training data
    :param test: test data
    :return: returns a model trained on the train data
    """
    if args.eo:
        m = mnist_logit.MnistLogit(out_dim=2, num_neurons=HIDDEN_LAYER_SIZE)
    else:
        m = mnist_logit.MnistLogit(out_dim=10, num_neurons=HIDDEN_LAYER_SIZE)
    m.train_model(train, test, iters=4000, verbose=False)
    return m

def swap_and_test_layer(m1, m2, test):
    """
    Replaces the first layer of m1 with the first layer of m2 and evaluates the results. Then reverts m1 to
    its original first layer.
    :param m1: Model 1
    :param m2: Model 2
    :param test: Data to evaluate on
    :return:
    """
    m1_w1 = m1.get_w1()
    m1_b1 = m1.get_b1()
    m1.set_w1(m2.get_w1())
    m1.set_b1(m2.get_b1())
    print "Swapped\t",
    m1.evaluate(test)
    m1.set_w1(m1_w1)
    m1.set_b1(m1_b1)

def baseline_diff(test, m1, m2):
    """
    Train two models from different starting points. Replace the first layer from the first model with
    the first layer from the second model. Evaluate the resulting model.
    """
    print "Running baseline_diff (layer swap)..."
    swap_and_test_layer(m1, m2, test)

def average_first_layer(test, m1, m2):
    m1_w1 = m1.get_w1()
    m1_b1 = m1.get_b1()
    m1.set_w1((m1_w1 + m2.get_w1()) / 2.0)
    m1.set_b1((m1_b1 + m2.get_b1()) / 2.0)
    l, a = m1.evaluate(test, verbose=False)
    m1.set_w1(m1_w1)
    m1.set_b1(m1_b1)
    return l, a


def swap_neuron(m1, m2, n1, n2):
    """

    Takes the n2th neuron in m2 and inserts in place of the n1th neuron in m1.
    :param m1: Destination model
    :param m2: Target model
    :param n1: destination neuron
    :param n2: target neuron
    """
    new_w = m1.get_w1()
    new_w[:, n1] = m2.get_w1()[:, n2]
    new_b = m1.get_b1()
    new_b[n1] = m2.get_b1()[n2]
    m1.set_w1(new_w)
    m1.set_b1(new_b)
    return m1


def average_neuron(m1, m2, n1, n2):
    """

    Takes the n2th neuron in m2 and inserts in place of the n1th neuron in m1.
    :param m1: Destination model
    :param m2: Target model
    :param n1: destination neuron
    :param n2: target neuron
    """
    new_w = m1.get_w1()
    new_w[:, n1] = (new_w[:, n1] + m2.get_w1()[:, n2]) / 2.0
    new_b = m1.get_b1()
    new_b[n1] = (new_b[n1] + m2.get_b1()[n2]) / 2.0
    m1.set_w1(new_w)
    m1.set_b1(new_b)
    return m1

def swap_and_test_neuron(m_dest, m_source, n_dest, n_source, data, verbose=False):
    """
    Takes the n_source neuron in m_source and inserts in place of the n_dest neuron in m_dest. Evaluates the resuls
    and reverts m1.
    :param m_dest: Destination model
    :param m_source: Source model
    :param n_dest: destination neuron
    :param n_source: Source neuron
    :return:
    """
    if verbose:
        print "Inserting N-%d in M1 at N-%d in M2" % (n_source, n_dest)
    w_orig = m_dest.get_w1()
    b_orig = m_dest.get_b1()
    new_w = m_dest.get_w1()
    new_w[:, n_dest] = m_source.get_w1()[:, n_source]
    new_b = m_dest.get_b1()
    new_b[n_dest] = m_source.get_b1()[n_source]
    m_dest.set_w1(new_w)
    m_dest.set_b1(new_b)
    loss, accuracy = m_dest.evaluate(data, verbose=verbose)
    m_dest.set_w1(w_orig)
    m_dest.set_b1(b_orig)
    return loss, accuracy


def loss_align_neurons(m_dest, m_source, test, average = False):
    """
    Calculate the loss of all neuron swap-pairs, and use the Hungaria algorithm to determine the optimal
    swaps.
    :param m_dest: Model with weights to replace (destination)
    :param m_source: Model with weights to use (source)
    :param average: if true, the final model will contain the averaged neurons. Otherwise, the final model will
    contain the swapped neurons.
    :param test: data to calculate loss on
    :return: a new model with swapped neurons
    """
    #new_model = load_model("m1")
    new_model = copy_model(m_dest)

    costs = np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))

    for n_dest in range(HIDDEN_LAYER_SIZE):
        for n_source in range(HIDDEN_LAYER_SIZE):
            loss, accuracy = swap_and_test_neuron(m_dest=new_model, m_source=m_source, n_dest=n_dest,
                                 n_source=n_source, data=test)
            costs[n_dest, n_source] = 1.0-accuracy

    dest_indices, source_indices = linear_sum_assignment(costs)

    for dest, source in zip(dest_indices, source_indices):
        if average:
            new_model = average_neuron(new_model, m_source, dest, source)
            loss, accuracy = new_model.evaluate(test, verbose=False)
            logger.info("Averaging %d (source) with %d (destination). Loss = %f Accuracy = %f" %
                        (source, dest, loss, accuracy))
        else:
            new_model = swap_neuron(new_model, m_source, dest, source)
            loss, accuracy = new_model.evaluate(test, verbose=False)
            logger.info("Inserting %d (source) for %d (destination). Loss = %f Accuracy = %f" %
                        (source, dest, loss, accuracy))

    return new_model


def covariance_alignment(m_dest, m_source, test, average = False):
    """
       Align neurons with a one-to-one correspondance based on covariance. Use the hungarian algorithm to sort the
       swaps.

       :param m_dest: Model with weights to replace (destination)
       :param m_source: Model with weights to use (source)
       :param average: if true, the final model will contain the averaged neurons. Otherwise, the final model will
       contain the swapped neurons.
       :param test: data to evaluate on
       :return:
    """
    new_model = load_model("m1")

    costs = np.zeros((HIDDEN_LAYER_SIZE, HIDDEN_LAYER_SIZE))

    for n_dest in range(HIDDEN_LAYER_SIZE):
        for n_source in range(HIDDEN_LAYER_SIZE):
            dest_w = np.append(m_dest.get_w1()[:, n_dest], m_dest.get_b1()[n_dest])
            source_w = np.append(m_source.get_w1()[:,n_source], m_source.get_b1()[n_source])
            cov = np.cov(dest_w, source_w)[0, 1]
            costs[n_dest, n_source] = 0.0-cov

    dest_indices, source_indices = linear_sum_assignment(costs)

    for dest, source in zip(dest_indices, source_indices):
        if average:
            new_model = average_neuron(new_model, m_source, dest, source)
            loss, accuracy = new_model.evaluate(test, verbose=False)
            logger.info("Averaging %d (source) with %d (destination). Loss = %f Accuracy = %f" %
                        (source, dest, loss, accuracy))
        else:
            new_model = swap_neuron(new_model, m_source, dest, source)
            loss, accuracy = new_model.evaluate(test, verbose=False)
            logger.info("Inserting %d (source) for %d (destination). Loss = %f Accuracy = %f" %
                        (source, dest, loss, accuracy))

    return new_model


def main():
    a1_train, a1_test, a2_train, a2_test = init(agent1=[0, 1, 2, 3, 4], agent2=[5, 6, 7, 8, 9])
    if args.eo:
        logger.info("Launching even/odd task")
    else:
        logger.info("Launching digit recognition task")

    if args.new_model:
        logger.info("Creating new model.")
        m1 = get_new_random_model(a1_train, a1_test)
        m2 = get_new_random_model(a2_train, a2_test)
        save_model(m1, "m1")
        save_model(m2, "m2")
    else:
        logger.info("Loading old model from alignment/")
        m1 = load_model("m1")
        m2 = load_model("m2")


    l, a = m1.evaluate(a1_test, verbose=False)
    logger.info("M1 x a1\tloss=%f\taccuracy=%f" % (l, a))
    l, a = m2.evaluate(a2_test, verbose=False)
    logger.info("M2 x a1\tloss=%f\taccuracy=%f" % (l, a))

    l, a = average_first_layer(a1_test, m1, m2)
    logger.info("Averaged m1 first layer on a1_test\tloss=%f\taccuracy=%f" % (l, a))
    l, a = average_first_layer(a2_test, m1, m2)
    logger.info("Averaged m1 first layer on a2_test\tloss=%f\taccuracy=%f" % (l, a))
    logger.info("Realigning models")
    realigned_model = loss_align_neurons(m1, m2, a1_test, average=True)
    l, a = realigned_model.evaluate(a2_test, verbose=False)
    logger.info("Realigned averaged m1 first layer on a2\tloss=%f\taccuracy=%f" % (l, a))
    l, a = realigned_model.evaluate(a1_test, verbose=False)
    logger.info("Realigned averaged m1 first layer on a1\tloss=%f\taccuracy=%f" % (l, a))



if __name__ == '__main__':
    main()
