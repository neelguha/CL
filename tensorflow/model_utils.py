"""
Useful functions for saving/loading models from file.
"""

import os
import numpy as np

def save_model(model, model_name):
    output_dir = "mt_models/%s_%d" % (model_name, model.num_hidden)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save("%s/w1" % output_dir, model.get_w1())
    np.save("%s/b1" % output_dir, model.get_b1())
    np.save("%s/w2" % output_dir, model.get_w2())
    np.save("%s/b2" % output_dir, model.get_b2())


def load_model(model_name, num_hidden, model_type):
    model_dir = "mt_models/%s_%d" % (model_name, num_hidden)
    w1 = np.load("%s/w1.npy" % model_dir)
    b1 = np.load("%s/b1.npy" % model_dir)
    w2 = np.load("%s/w2.npy" % model_dir)
    b2 = np.load("%s/b2.npy" % model_dir)
    new_model = model_type(num_hidden)
    new_model.set_vars(w1, b1, w2, b2)
    return new_model