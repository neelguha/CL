import tensorflow as tf
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from agent import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main():
    a0 = Agent(0, False)
    opt = a0.get_optimum()
    print a0.get_loss_accuracy(opt)
    w, b = a0.get_gradient().split()
    print w.tolist()
    print b
    '''for _ in range(1000):
        new_w = opt.permute(var=5.0)
        w, b = new_w.split()
        local_cost, local_accuracy = a0.get_loss_accuracy(w, b, use_test=False)
        print local_accuracy, local_cost, w[:5,0]'''

if __name__ == '__main__':
    main()