import tensorflow as tf
from model_types.mnist_second_layer import *
from data import *
import sys
import copy
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


INPUT_SIZE = 20
WINDOW_WIDTH = 20

def main():

    #full_train, full_test = load_full_data()
    d0_train, d0_test = load_transformed_digit_data(digit=0, input_size=20)
    d1_train, d1_test = load_transformed_digit_data(digit=1, input_size=20)

    d_all_train = copy.deepcopy(d0_train)
    d_all_test = copy.deepcopy(d0_test)
    d_all_train.add_data(d1_train.images, d1_train.labels)
    d_all_test.add_data(d1_test.images, d1_test.labels)

    m0 = PartialModel(INPUT_SIZE, WINDOW_WIDTH)
    _, m1_base_loss = m0.train_model(d0_train, d0_test, iters=400)
    m1 = PartialModel(INPUT_SIZE, WINDOW_WIDTH)
    _, m2_base_loss = m1.train_model(d1_train, d1_test, iters=400)
    m_all = PartialModel(INPUT_SIZE, WINDOW_WIDTH)
    m_all.train_model(d_all_train, d_all_test, iters=400)

    print "M_gold on combined:", m_all.evaluate(d_all_test)
    print "M_0 on combined:", m0.evaluate(d_all_test)
    print "M_1 on combined:", m1.evaluate(d_all_test)

    m0.vary_vars_b(d0_test)
    m1.vary_vars_b(d1_test)

    m0.vary_vars_w(d0_test)
    m1.vary_vars_w(d1_test)

    models = [m0, m1]
    new_w = merge_models_w(models=models, input_size=INPUT_SIZE, window_width=WINDOW_WIDTH)
    new_b = merge_models_b(models, window_width=WINDOW_WIDTH)
    m0.set_w(new_w)
    m0.set_b(new_b)
    accuracy, loss = m0.evaluate(d_all_test)
    print "M_new on combined: Accuracy=%f\tLoss=%f" % (accuracy, loss)








if __name__ == '__main__':
    main()


