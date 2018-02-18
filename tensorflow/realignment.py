""" Code for realigning two models """
import numpy as np
from scipy.optimize import linear_sum_assignment


def average_models(m1, m2):
    new_m = m1.copy()
    new_w1 = (new_m.get_w1() + m2.get_w1()) / 2.0
    new_b1 = (new_m.get_b1() + m2.get_b1()) / 2.0
    new_w2 = (new_m.get_w2() + m2.get_w2()) / 2.0
    new_b2 = (new_m.get_b2() + m2.get_b2()) / 2.0
    new_m.set_vars(w1=new_w1, b1=new_b1, w2=new_w2, b2=new_b2)
    return new_m

def switch_neurons(m1, m2, n1, n2):
    """
    Inserts neurons at n2 in m2 for neuron at n1 in m1
    :param m1:
    :param m2:
    :param n1:
    :param n2:
    :return:
    """
    # Rearrange hidden layer weights
    new_w = m1.get_w1()[:]
    new_w[:, n1] = m2.get_w1()[:, n2]
    new_b = m1.get_b1()[:]
    new_b[n1] = m2.get_b1()[n2]
    m1.set_vars(w1=new_w, b1=new_b)

    # Rearrange final layer weights
    new_w = m1.get_w2()[:]
    new_w[n1,:] = m2.get_w2()[n2,:]
    m1.set_vars(w2=new_w)


def realign(model1, model2, method, data = None):
    """
    Creates a copy of model2 and realigns it to model1.
    :param model1:
    :param model2:
    :param method:
    :return:
    """
    realigned_m2 = model2.copy()
    if method == "cov":
        m1_indices, m2_indices = covariance_align(model1, model2)
    elif method == "loss":
        if data == None:
            raise ValueError("No data passed for loss-alignment")
        m1_indices, m2_indices = loss_align(model1, model2, data)
    else:
        raise ValueError("oops")


    # Observe that we make the changes to new_model, not model2
    sorted_m2_indices = m2_indices[np.argsort(m1_indices)]
    sorted_m1_indices = sorted(m1_indices)

    for m1_i, m2_i in zip(sorted_m1_indices, sorted_m2_indices):
        switch_neurons(realigned_m2, model2, m1_i, m2_i)

    return realigned_m2


def loss_align(model1, model2, data):
    costs = np.zeros((model1.num_hidden, model1.num_hidden))

    for n_dest in range(model1.num_hidden):
        for n_source in range(model1.num_hidden):
            loss, accuracy = swap_and_test_neuron(m_dest=model1, m_source=model2, n_dest=n_dest,
                                                  n_source=n_source, data=data)
            costs[n_dest, n_source] = 1.0 - loss

    m1_neurons, m2_neurons= linear_sum_assignment(costs)

    return m1_neurons, m2_neurons


def covariance_align(model1, model2):
    costs = np.zeros((model1.num_hidden, model1.num_hidden))

    for n_dest in range(model1.num_hidden):
        for n_source in range(model1.num_hidden):
            dest_w = np.append(model1.get_w1()[:, n_dest], model1.get_b1()[n_dest])
            source_w = np.append(model2.get_w1()[:, n_source], model2.get_b1()[n_source])
            cov = np.cov(dest_w, source_w)[0, 1]
            costs[n_dest, n_source] = 0.0 - cov

    m1_neurons, m2_neurons= linear_sum_assignment(costs)
    return m1_neurons, m2_neurons




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
    w1_orig = m_dest.get_w1()
    b1_orig = m_dest.get_b1()
    w2_orig = m_dest.get_w2()

    '''new_w1 = m_dest.get_w1()
    new_w1[:, n_dest] = m_source.get_w1()[:, n_source]
    new_b1 = m_dest.get_b1()
    new_b1[n_dest] = m_source.get_b1()[n_source]

    new_w2 = m_dest.get_w2()[:]
    new_w2[n_dest, :] = m_source.get_w2()[n_source, :]

    m_dest.set_vars(w1=new_w1, b1=new_b1, w2=new_w2)'''
    switch_neurons(m_dest, m_source, n_dest, n_source)
    loss, accuracy = m_dest.evaluate(data)
    m_dest.set_vars(w1=w1_orig, b1=b1_orig, w2=w2_orig)
    return loss, accuracy