"""
Experiment Version 4:
Given two models, first realign the models, then perform fisher based mapping to combine the models

"""
import tensorflow as tf
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




def combine(m1, m2, dc):
    m1.compute_fisher(dc.a1.validation.images)
    m2.compute_fisher(dc.a2.validation.images)

    solver = Solver(m1, m2)
    combined_model = solver.get_new_model()
    a, l = combined_model.evaluate(dc.all.test)
    logger.info("Combined on all: Accuracy = %f Loss = %f" % (a, l))



def main():

    # Fetch datasets corresponding to agent1 and agent2
    dc = DataController(EXP_TYPE, agent1=[0, 1, 2, 3, 4], agent2 = [5, 6, 7, 8, 9])
    logger.info("%d total train, %d total test" % (dc.all.train.get_count(), dc.all.test.get_count()))
    # Build model m1 and m2 on agent1 and agent2 data
    m1 = MnistEO(FLAGS.hidden, logger)
    logger.info("Training model 1...")
    m1.train_model(dc.a1.train, dc.a1.test)

    m2 = MnistEO(FLAGS.hidden, logger)
    logger.info("Training model 2...")
    m2.train_model(dc.a2.train, dc.a2.test)
    logger.info("Training gold model...")
    m_gold = MnistEO(FLAGS.hidden, logger)
    m_gold.train_model(dc.all.train, dc.all.test)

    a, l = m1.evaluate(dc.all.test)
    logger.info("M1 on all: Accuracy = %f Loss = %f" % (a, l))
    a, l = m2.evaluate(dc.all.test)
    logger.info("M2 on all: Accuracy = %f Loss = %f" % (a, l))

    avg_model = average_models(m1, m2)
    a, l = avg_model.evaluate(dc.a2.test)
    logger.info("M_avg Accuracy=%f Loss=%f" % (a, l))

    logger.info("Realigning models.")
    realigned_m2 = realign(m1,m2, method="cov")

    avg_model = average_models(m1, realigned_m2)
    a, l = avg_model.evaluate(dc.a2.test)
    logger.info("Realigned M_avg Accuracy=%f Loss=%f" % (a, l))
    # Compute fisher information
    combine(m1, realigned_m2, dc)
    logger.info("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    main()