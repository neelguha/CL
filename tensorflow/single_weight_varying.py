import tensorflow as tf
import argparse
import sys
from sklearn.model_selection import train_test_split
import os
import numpy as np
import random
from model_types.mnist_logit import *
from agent import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def main():
    m = MnistLogit(0)
    m.train_model()
    print m.count_number_trainable_params()
    print m.get_accuracy_cost()
    # Generate starting weight parameters
    # Create Agent 0
    # Create Agent 1

    # For each agent, iterate through each variable
        # Sample variable at random points
        # Record sensitivity at different points
        # Establish "valid" range

    # Intersect ranges






if __name__ == '__main__':
    main()