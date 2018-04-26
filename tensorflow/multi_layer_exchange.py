from mt_model import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from agent_model import *
from controller import *
#from flexible_model import *
from tabulate import tabulate
import model_utils
from scipy.optimize import linear_sum_assignment


# Constants
AGENT1 = [0, 1, 2, 3, 4]
AGENT2 = [5, 6, 7, 8, 9]
NUM_OUTPUT = 10

# Command line flags initialization
flags = tf.app.flags
FLAGS = flags.FLAGS

def run_tests(mt, digits1, digits2, gold):

    table = []
    accuracy, loss = gold.evaluate(mt.all_test)
    table.append(["Gold", accuracy, loss])
    accuracy, loss = digits1.evaluate(mt.all_test)
    table.append(["Digits1", accuracy, loss])
    accuracy, loss = digits2.evaluate(mt.all_test)
    table.append(["Digits2", accuracy, loss])
    # Average models
    avg_m = model_utils.average_models(digits1, digits2)
    accuracy, loss = avg_m.evaluate(mt.all_test)
    table.append(["Averaged", accuracy, loss])
    print(tabulate(table, headers=["Baseline Performance", "Accuracy", "Loss"], tablefmt="fancy_grid"))

def new_models(mt):
    print("Creating new models")
    w1 = np.random.normal(0.0, 0.1, (784, FLAGS.hidden))
    b1 = np.random.normal(0.0, 0.1, (FLAGS.hidden))
    w2 = np.random.normal(0.0, 0.1, (FLAGS.hidden, NUM_OUTPUT))
    b2 = np.random.normal(0.0, 0.1, (NUM_OUTPUT))
    architecture = [FLAGS.hidden, NUM_OUTPUT]
    gold = AgentModel(architecture)
    gold.set_vars([(w1, b1), (w2, b2)])
    print("Training gold...")
    gold.train_model(mt.all_train, mt.all_test)
    model_utils.save_model(gold, "gold")

    digits1 = AgentModel(architecture)
    digits1.set_vars([(w1, b1), (w2, b2)])
    print("Training mnist1 digits...")
    digits1.train_model(mt.mnist1_train, mt.mnist1_test)
    model_utils.save_model(digits1, "digit1")

    digits2 = AgentModel(architecture)
    digits2.set_vars([(w1, b1), (w2, b2)])
    print ("Training mnist2 digits...")
    digits2.train_model(mt.mnist2_train, mt.mnist2_test)
    model_utils.save_model(digits2, "digit2")
    return digits1, digits2, gold

def main():

    mt = MTData()
    architecture = [FLAGS.hidden, NUM_OUTPUT]
    print
    if FLAGS.new:
        digits1, digits2, gold = new_models(mt)
    else:
        digits1 = model_utils.load_model_with_arch("digit1",  architecture, AgentModel)
        digits2 = model_utils.load_model_with_arch("digit2", architecture, AgentModel)
        gold = model_utils.load_model_with_arch("gold", architecture, AgentModel)

    c = CombinedModel()

    table = []
    table.append(["Agent Hidden Nodes", FLAGS.hidden])
    table.append(["Epsilon", FLAGS.epsilon])
    table.append(["Controller Dataset size", FLAGS.v_size])
    table.append(["Delta", FLAGS.delta])
    table.append(["Agent Sample", FLAGS.agent_sample])
    print(tabulate(table, headers=["Parameter", "Value"], tablefmt="fancy_grid"))

    d1_sample = mt.mnist1_train.next_batch(FLAGS.agent_sample)
    d2_sample = mt.mnist2_train.next_batch(FLAGS.agent_sample)
    m1_spaces = []
    m2_spaces = []

    for h in range(FLAGS.hidden):
        m1_spaces.append(digits1.get_hidden_neuron_space(h, layer=0, data=d1_sample, epsilon=FLAGS.epsilon,
                                                         delta=FLAGS.delta))
        m2_spaces.append(digits2.get_hidden_neuron_space(h, layer=0, data=d2_sample, epsilon=FLAGS.epsilon,
                                                         delta=FLAGS.delta))

    costs = np.ones((FLAGS.hidden, FLAGS.hidden))
    for d1_n in range(FLAGS.hidden):
        for d2_n in range(FLAGS.hidden):
            dist = model_utils.get_full_neuron_distance(d1_n, d2_n, digits1, digits2)
            if dist < m1_spaces[d1_n] + m2_spaces[d2_n]:
                costs[d1_n, d2_n] = 0.0

    d1_neurons, d2_neurons = linear_sum_assignment(costs)
    for d1_n, d2_n in zip(d1_neurons, d2_neurons):
        dist = model_utils.get_full_neuron_distance(d1_n, d2_n, digits1, digits2)
        if dist < m1_spaces[d1_n] + m2_spaces[d2_n]:
            new_w_in, new_b, new_w_out = model_utils.average_hidden_neuron_radius_wise(digits1, digits2, d1_n, d2_n,
                                                                                       m1_spaces[d1_n], m2_spaces[d2_n])
            c.add_hidden_neuron(new_w_in, new_w_out, new_b)
        else:
            d1w1, d1b1 = digits1.get_parameters(0)
            d1w2, d1b2 = digits1.get_parameters(1)

            d2w1, d2b1 = digits2.get_parameters(0)
            d2w2, d2b2 = digits2.get_parameters(1)

            c.add_hidden_neuron(d1w1[:, d1_n], d1w2[d1_n, :], d1b1[d1_n])
            c.add_hidden_neuron(d2w1[:, d2_n], d2w2[d2_n, :], d2b1[d2_n])

    for o_n in range(10):
        _, d1b2 = digits1.get_parameters(1)
        _, d2b2 = digits2.get_parameters(1)
        c.add_output_bias(o_n, 0.5 * d1b2[o_n] + 0.5 * d2b2[o_n])

    table = []
    accuracy, loss = digits1.evaluate(mt.mnist1_test)
    table.append(["Digits1 on Mnist1", accuracy, loss])
    accuracy, loss = digits2.evaluate(mt.mnist2_test)
    table.append(["Digits2 on Mnist2", accuracy, loss])
    print(tabulate(table, headers=["Local Agent Performance", "Accuracy", "Loss"], tablefmt="fancy_grid"))

    run_tests(mt, digits1, digits2, gold)

    print("%d total neurons in aggregate model" % c.hidden)

    table = []
    new_model = c.convert_to_model()
    accuracy, loss = new_model.evaluate(mt.all_test)
    table.append(["Aggregate", accuracy, loss])

    full_d_sample = mt.mnist1_train.next_batch(FLAGS.v_size / 2)
    full_d_sample.add_data_obj(mt.mnist2_train.next_batch(FLAGS.v_size / 2))
    new_model.train_second_layer(full_d_sample, mt.all_test, iters=2000)
    accuracy, loss = new_model.evaluate(mt.all_test)
    table.append(["Tuned Aggregate", accuracy, loss])

    digits1.train_model(full_d_sample, mt.all_test)
    accuracy, loss = digits1.evaluate(mt.all_test)
    table.append(["Tuned Digits1", accuracy, loss])
    print(tabulate(table, headers=["Exp", "Accuracy", "Loss"], tablefmt="fancy_grid"))

    # Assume each model contains two layers

    # initialize m1, m2 with 1 hidden layers
    # Train m1 on d1, m2 on d2

    # identify first intersection layer for m1, m2

    # transform d1 based on layer
    # transform d2 based on layer

    # reinitialize m1, m2 with no hidden layers
    # identify intersection for final layer for m1, m2






    pass








if __name__ == '__main__':
    flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
    flags.DEFINE_integer("layers", 1, "Number of hidden layers")
    flags.DEFINE_integer("hidden", 50, "Number of units in hidden layer 1.")
    flags.DEFINE_bool("new", False, "Whether to build new models")
    flags.DEFINE_float("epsilon", 1.0, "Epsilon Value")
    flags.DEFINE_float("delta", 1.0, "Delta Value")
    flags.DEFINE_integer("v_size", 200, "Number of random images for aggregate fine-tuning")
    flags.DEFINE_integer("agent_sample", 200, "Number of samples to calculate loss on ")
    main()