"""
Alignment v2:

1. Measure importance of each neuron (using standard deviation on sample of outputs)

2. Determine which neurons are "close"

"""
from eo_two_layer import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging, coloredlogs
from controller import *
from realignment import *
import time
start_time = time.time()


# Constants
AGENT1 = [0, 1, 2, 3, 4]
AGENT2 = [5, 6, 7, 8, 9]
EXP_TYPE = "eo" # Even-Odd

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
flags.DEFINE_string("align", "loss", "Method to align networks")


def rank_neurons(model, data):
    vals = model.get_neuron_values(data)
    vars = np.var(vals,axis=0)
    means = np.mean(vals, axis=0)
    sorted_neurons = np.flip(np.argsort(vars), 0)
    for index in sorted_neurons:
        logger.info("Neuron-%d: Var = %f\tMean = %f" % (index, vars[index], means[index]))

def align_neurons(m1, m2, data):
    # Shape = (num_samples, num_hidden)
    m1_vals = m1.get_neuron_values(data)
    m1_vals = m1_vals / np.linalg.norm(m1_vals, axis=0)
    m2_vals = m2.get_neuron_values(data)
    m2_vals = m2_vals / np.linalg.norm(m2_vals, axis=0)
    #TODO: do we need to norm the neural outputs for each data point?

    for n1 in range(FLAGS.hidden):
        distances = []
        for n2 in range(FLAGS.hidden):
            #dist = np.dot(m1_vals[:, n1], m2_vals[:, n2])
            #dist = -np.sqrt(np.sum(np.square(m1_vals[:, n1] - m2_vals[:, n2])))
            dist = np.cov(m1_vals[:, n1], m2_vals[:, n2])[0,1]
            distances.append(dist)
        matched_neuron = np.argmax(distances)
        best_dist = max(distances)
        logger.info("(M1-N-%d, M2-N-%d): Dot = %f " % (n1, matched_neuron, best_dist))

def main():

    # Fetch datasets corresponding to agent1 and agent2
    dc = DataController(EXP_TYPE, agent1=[0, 1, 2, 3, 4], agent2 = [5, 6, 7, 8, 9])
    logger.info("%d total train, %d total test" % (dc.all.train.get_count(), dc.all.test.get_count()))
    # Build model m1 and m2 on agent1 and agent2 data
    m1 = MnistEO(FLAGS.hidden, logger)
    logger.info("Training model 1 with %d hidden neurons..." % FLAGS.hidden)
    m1.train_model(dc.a1.train, dc.a1.test)
    m2 = MnistEO(FLAGS.hidden, logger)
    logger.info("Training model 2 with %d hidden neurons..." % FLAGS.hidden)
    m2.train_model(dc.a2.train, dc.a2.test)
    logger.info("Training gold model...")
    m_gold = MnistEO(FLAGS.hidden, logger)
    m_gold.train_model(dc.all.train, dc.all.test)

    a1_sample = dc.sample_by_agent(1, 100)
    a2_sample = dc.sample_by_agent(2, 100)

    print m_gold.evaluate(dc.a1.test)
    print m_gold.evaluate(dc.a2.test)



if __name__ == '__main__':
    main()