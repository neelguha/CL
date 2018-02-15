from tensorflow.examples.tutorials.mnist import input_data
from model_types import mnist_logit
from scipy import spatial
from data import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse

HIDDEN_LAYER_SIZE = 20 # The input size for the second layer

parser = argparse.ArgumentParser(description='MNIST Alignment (Tensorflow)')
parser.add_argument("--new_model", default=False, action="store_true" , help="Flag to do something")
args = parser.parse_args()


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
        even_odd = digit % 2
        one_hot = np.zeros(2)
        one_hot[even_odd] = 1.0
        if digit in agent1:
            a1_labels.append(one_hot)
            a1_data.append(image)
        elif digit in agent2:
            a2_labels.append(one_hot)
            a2_data.append(image)

    return Data(np.array(a1_data), np.array(a1_labels)), Data(np.array(a2_data), np.array(a2_labels))


def init(agent1, agent2):
    """
    :return: (Train data, test data)
    """
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    a1_train, a2_train = sort_data(agent1, agent2, mnist.train.images, mnist.train.labels)
    a1_test, a2_test = sort_data(agent1, agent2, mnist.test.images, mnist.test.labels)
    print "Sorted data."
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
    return mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE, out_dim=2, w1=w1, w2=w2, b1=b1, b2=b2)

def get_new_random_model(train, test):
    """
    :param train: training data
    :param test: test data
    :return: returns a model trained on the train data
    """
    m = mnist_logit.MnistLogit(out_dim=2, num_neurons=HIDDEN_LAYER_SIZE)
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

def baseline_diff(train, test, m1 = None, m2 = None):
    """
    Train two models from different starting points. Replace the first layer from the first model with
    the first layer from the second model. Evaluate the resulting model.
    """
    print "Running baseline_diff..."
    if m1 == None:
        print "Training new m1"
        m1 = mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE)
        print "M1\t",
        m1.train_model(train, test,verbose=False)
    if m2 == None:
        print "Training new m2"
        m2 = mnist_logit.MnistLogit(num_neurons=HIDDEN_LAYER_SIZE)
        print "M2\t",
        m2.train_model(train, test, verbose=False)
    swap_and_test_layer(m1, m2, test)

def baseline_same(train, test):
    """
    Train two models from the same starting point. Replace the first layer from the first model with
    the first layer from the second model. Evaluate the resulting model.
    """
    print "Running baseline_same..."
    w1 = np.random.normal(0, 1.0, (784, HIDDEN_LAYER_SIZE))
    b1 = np.random.normal(0, 1.0, (HIDDEN_LAYER_SIZE))
    w2 = np.random.normal(0, 1.0, (HIDDEN_LAYER_SIZE, 10))
    b2 = np.random.normal(0, 1.0, (10))
    m1 = mnist_logit.MnistLogit(HIDDEN_LAYER_SIZE, w1, w2, b1, b2)
    print "M1\t",
    m1.train_model(train, test,verbose=False)
    m2 = mnist_logit.MnistLogit(HIDDEN_LAYER_SIZE, w1, w2, b1, b2)
    print "M2\t",
    m2.train_model(train, test, verbose=False)
    swap_and_test_layer(m1, m2, test)



def swap_neuron(m1, m2, n1, n2, test):
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


def align_neurons(m_dest, m_source, test):
    """
    Align neurons with a one-to-one correspondance

    :param m_dest: Model with weights to replace (destination)
    :param m_source: Model with weights to use (source)
    :param test:
    :return:
    """

    new_model = load_model("m1")
    replaced_neurons = [] # neurons in m_dest that have already been replaced
    sourced_neurons = [] # neurons in m_source that have already been used to replace
    dest_source_map = {} # map of destination index to source index

    # Replacing every neuron in m_dest will take HIDDEN_LAYER_SIZE iterations. At each iteration, we
    for iter in range(HIDDEN_LAYER_SIZE):
        scores = []
        pairs = []
        for n_dest in range(HIDDEN_LAYER_SIZE):
            if n_dest in replaced_neurons:
                continue
            for n_source in range(HIDDEN_LAYER_SIZE):
                if n_source in sourced_neurons:
                    continue
                loss, accuracy = swap_and_test_neuron(m_dest=new_model, m_source=m_source, n_dest=n_dest,
                                                      n_source=n_source, data=test)
                scores.append(accuracy)
                pairs.append((n_dest, n_source))

        final_dn, final_sn = pairs[np.argmax(scores)]
        sourced_neurons.append(final_sn)
        replaced_neurons.append(final_dn)
        dest_source_map[final_dn] = final_sn
        new_model = swap_neuron(new_model, m_source, final_dn, final_sn, test)
        loss, accuracy = new_model.evaluate(test, verbose=False)
        print
        print "Inserting %d (source) for %d (destination). Loss = %f Accuracy = %f" % (final_sn, final_dn,loss, accuracy)

    print "New model performance: ",
    new_model.evaluate(test, verbose=True)
    return new_model

def main():
    a1_train, a1_test, a2_train, a2_test = init(agent1=[0, 1, 2, 3], agent2=[4, 5, 6, 7])
    if args.new_model:
        m1 = get_new_random_model(a1_train, a1_test)
        m2 = get_new_random_model(a2_train, a2_test)
        save_model(m1, "m1")
        save_model(m2, "m2")
    else:
        m1 = load_model("m1")
        m2 = load_model("m2")
    print "M1\t",
    m1.evaluate(a1_test)
    print "M2\t",
    m2.evaluate(a2_test)
    print
    baseline_diff(a1_train, a1_test, m1, m2)
    print "Realigning models"
    realigned_model = align_neurons(m1, m2, a1_test)




if __name__ == '__main__':
    main()
