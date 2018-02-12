""" FUNCTIONAL DECOMPOSITION EXPERIMENTS """

from data import *
from model_types.mnist_second_layer import *
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
from termcolor import colored


INPUT_SIZE = 20
DIGIT_OUTPUT = True
OUTPUT_SIZE = 10
WINDOW_WIDTH = 20
EPSILON = 0.3
ITERS = 50
VERBOSE = False







def main():
    agent1 = Agent(output_digit=False, file_prefix="transformed_inputs/size_%d" % INPUT_SIZE, digits=[0, 1, 2, 3, 4])
    agent2 = Agent(output_digit=False, file_prefix="transformed_inputs/size_%d" % INPUT_SIZE, digits=[5, 6, 7, 8, 9])

    for _ in range(10):
        m1 = PartialModel(INPUT_SIZE, WINDOW_WIDTH, output_size=2)
        w_init = m1.get_w()
        b_init = m1.get_b()

        m1.train_model(agent1.train, agent1.test, iters=1000)
        w_final = m1.get_w()
        b_final = m1.get_b()
        diffs = np.abs(w_final - w_init)
        #print np.average(diffs, axis=1)
        print np.around(np.average(diffs, axis=1), decimals=2).tolist()
        diffs = np.abs(b_final - b_init)



if __name__ == '__main__':
    main()