"""
Epsilon Perturbation based on fisher information
"""

from data import *
from model_types.mnist_second_layer import *
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
from termcolor import colored
import tensorflow as tf

INPUT_SIZE = 20
DIGIT_OUTPUT = True
OUTPUT_SIZE = 10
WINDOW_WIDTH = 20
EPSILON = 0.3
ITERS = 1
VERBOSE = False

assert ((DIGIT_OUTPUT == True and OUTPUT_SIZE == 10) or (DIGIT_OUTPUT == False and OUTPUT_SIZE == 2))


class Solver():


    def __init__(self, c1_w, c1_b, c1_w_f, c1_b_f, c2_w, c2_b, c2_w_f, c2_b_f ):
        self.sess = tf.Session()
        self.W = tf.Variable(tf.truncated_normal(c1_w.shape, stddev=0.1), dtype=tf.float32)
        self.b = tf.Variable(tf.truncated_normal(c1_b.shape, stddev=0.1), dtype=tf.float32)
        self.loss = tf.multiply(tf.abs(self.W - c1_w), c1_w_f) + \
                    tf.multiply(tf.abs(self.W - c2_w), c2_w_f) + \
                    tf.multiply(tf.abs(self.b - c1_b), c1_b_f) + \
                    tf.multiply(tf.abs(self.b - c2_b), c2_b_f)
        self.loss = tf.reduce_sum(self.loss)
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        for _ in range(10000):
            self.sess.run(self.train_step)
            print "Loss: %f" % self.sess.run(self.loss)

    def get_w(self):
        return self.sess.run(self.W)

    def get_b(self):
        return self.sess.run(self.b)

def main():
    agent1 = Agent(output_digit=True, file_prefix="transformed_inputs/size_%d" % INPUT_SIZE, digits=[0, 1, 2, 3, 4])
    agent2 = Agent(output_digit=True, file_prefix="transformed_inputs/size_%d" % INPUT_SIZE, digits=[5, 6, 7, 8, 9])
    all_data = Agent(output_digit=True, file_prefix="transformed_inputs/size_%d" % INPUT_SIZE, digits=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    m_all = PartialModel(INPUT_SIZE, WINDOW_WIDTH, output_size=OUTPUT_SIZE)
    all_acc, all_loss = m_all.train_model(all_data.train, all_data.test, iters=1000)
    print "Gold Accuracy: %f\tGold Loss: %f" % (all_acc, all_loss)
    valid_iters = 0
    success = 0
    for _ in range(ITERS):
        print
        m1 = PartialModel(INPUT_SIZE, WINDOW_WIDTH, output_size=OUTPUT_SIZE)
        m1.train_model(agent1.train, agent1.test, iters=1000)
        m2 = PartialModel(INPUT_SIZE, WINDOW_WIDTH, output_size=OUTPUT_SIZE)
        m2.train_model(agent2.train, agent2.test, iters=1000)
        print "trained models"
        m1.compute_fisher(agent1.train.images)
        m2.compute_fisher(agent2.train.images)
        print "computed fisher"

        solver = Solver(m1.get_w(), m1.get_b(), m1.F_accum[0].astype(np.float32), m1.F_accum[1].astype(np.float32),
                        m2.get_w(), m2.get_b(), m2.F_accum[0].astype(np.float32), m2.F_accum[1].astype(np.float32))
        print "solved for new w/b"
        new_w = solver.get_w()
        new_b = solver.get_b()

        m1_all_acc, m1_all_loss = m1.evaluate(all_data.test)
        m2_all_acc, m2_all_loss = m2.evaluate(all_data.test)
        combined = PartialModel(INPUT_SIZE, WINDOW_WIDTH, output_size=OUTPUT_SIZE)
        combined.set_w(new_w)
        combined.set_b(new_b)


        print "M1 on all: Acc: %f\tLoss: %f" % (m1_all_acc, m1_all_loss)
        print "M2 on all: Acc: %f\tLoss: %f" % (m2_all_acc, m2_all_loss)

        passes = True
        accuracy, loss = combined.evaluate(agent1.test)
        if loss < EPSILON:
            print colored("M_new on Agent 1: Accuracy=%f\tLoss=%f" % (accuracy, loss), "green")
        else:
            passes = False
            print colored("M_new on Agent 1: Accuracy=%f\tLoss=%f" % (accuracy, loss), "red")
        accuracy, loss = combined.evaluate(agent2.test)
        if loss < EPSILON:
            print colored("M_new on Agent2: Accuracy=%f\tLoss=%f" % (accuracy, loss), "green")
        else:
            passes = False
            print colored("M_new on Agent2: Accuracy=%f\tLoss=%f" % (accuracy, loss), "red")
        if passes:
            valid_iters += 1
        accuracy, loss = combined.evaluate(all_data.test)
        if accuracy > m1_all_acc and accuracy > m2_all_acc:
            print colored("M_new on combined: Accuracy=%f\tLoss=%f" % (accuracy, loss), "green")
        else:
            print colored("M_new on combined: Accuracy=%f\tLoss=%f" % (accuracy, loss), "red")
        if accuracy > m1_all_acc and accuracy > m2_all_acc and passes:
            success += 1
    print "%d / %d" % (success, valid_iters)


if __name__ == '__main__':
    main()


