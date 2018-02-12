from data import *
from model_types.mnist_second_layer import *
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
from termcolor import colored


INPUT_SIZE = 50
DIGIT_OUTPUT = True
OUTPUT_SIZE = 10
WINDOW_WIDTH = 20
EPSILON = 0.3
ITERS = 50
VERBOSE = False

assert ((DIGIT_OUTPUT == True and OUTPUT_SIZE == 10) or (DIGIT_OUTPUT == False and OUTPUT_SIZE == 2))


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
        m1_all_acc, m1_all_loss = m1.evaluate(all_data.test)
        m2_all_acc, m2_all_loss = m2.evaluate(all_data.test)
        m1.vary_vars_b(agent1.train)
        m2.vary_vars_b(agent2.train)
        m1.vary_vars_w(agent1.train)
        m2.vary_vars_w(agent2.train)

        models = [m1, m2]
        new_w = merge_models_w(models=models, input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, window_width=WINDOW_WIDTH)
        new_b = merge_models_b(models, window_width=WINDOW_WIDTH, output_size=OUTPUT_SIZE)
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


