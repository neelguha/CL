"""
In order to simplify the structure of the code, any experiment will rely on this code here.

"""
import os
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
from data import *
from scipy import io as spio
from copy import deepcopy


'''
This is the basic type for all of our data objects
'''
class Data:
    ''' Represents a dataset.'''
    def __init__(self, images=[], labels=[]):
        self.images = images
        self.labels = labels


    def add_data(self, new_images, new_labels):
        if len(self.images) == 0:
            self.images = new_images
            self.labels = new_labels
        else:
            self.images = np.append(self.images, new_images, axis=0)
            self.labels = np.append(self.labels, new_labels, axis=0)

    def add_data_obj(self, new_data):
        self.images = np.append(self.images, new_data.images, axis=0)
        self.labels = np.append(self.labels, new_data.labels, axis=0)

    def next_batch(self, n):
        if n > len(self.images):
            raise ValueError("Batch size exceeds data")
        indices = np.random.choice(list(range(len(self.images))), int(n))
        return Data(self.images[indices], self.labels[indices])

    def get_count(self):
        return len(self.images)

class Agent:

    def __init__(self, type, train_images, train_labels, test_images, test_labels, validation_images,
                 validation_labels):
        if type == "eo":
            self.train = Data(train_images, self.even_odd_labels(train_labels))
            self.test = Data(test_images, self.even_odd_labels(test_labels))
            self.validation = Data(validation_images, self.even_odd_labels(validation_labels))
        elif type == "digit":
            self.train = Data(train_images, train_labels)
            self.test = Data(test_images, test_labels)
            self.validation = Data(validation_images, validation_labels)

    def even_odd_labels(self, labels):
        new_labels = []
        for label in labels:
            digit = np.argmax(label)
            if digit % 2 == 0:
                new_labels.append([1, 0])
            else:
                new_labels.append([0, 1])
        return np.array(new_labels)

class DataController:

    def __init__(self, type, agent1=[], agent2=[]):
        """
        :param type: Could be "eo" (even/odd) or "digit" (digit recognition)
        """
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        a1_train_images, a1_train_labels, a2_train_images, a2_train_labels = self.sort_data(agent1, agent2,
                                                                                            mnist.train.images,
                                                                                            mnist.train.labels)
        a1_test_images, a1_test_labels, a2_test_images, a2_test_labels = self.sort_data(agent1, agent2,
                                                                                        mnist.test.images,
                                                                                        mnist.test.labels)
        a1_validation_images, a1_validation_labels, a2_validation_images, a2_validation_labels = \
            self.sort_data(agent1, agent2, mnist.validation.images, mnist.validation.labels)

        self.a1 = Agent(type, a1_train_images, a1_train_labels, a1_test_images, a1_test_labels, a1_validation_images,
                        a1_validation_labels)
        self.a2 = Agent(type, a2_train_images, a2_train_labels, a2_test_images, a2_test_labels, a2_validation_images,
                        a2_validation_labels)
        self.all = Agent(type, mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels,
                         mnist.validation.images, mnist.validation.labels)

    def sort_data(self, agent1, agent2, images, labels):
        """
        :param agent1: Digits learned by agent 1
        :param agent2: Digits learned by agent 2
        :param images:
        :param labels:
        :return:
        """
        a1_data = []
        a1_labels = []
        a2_data = []
        a2_labels = []

        data = zip(labels, images)
        for label, image in data:
            digit = np.argmax(label)
            if digit in agent1:
                a1_labels.append(label)
                a1_data.append(image)
            elif digit in agent2:
                a2_labels.append(label)
                a2_data.append(image)
        return np.array(a1_data), np.array(a1_labels), np.array(a2_data), np.array(a2_labels)

    def sample_by_agent(self, agent, count):
        if agent == 1:
            sample_indices = np.random.choice(range(len(self.a1.validation.images)), count, replace=False)
            return Data(self.a1.validation.images[sample_indices], self.a1.validation.labels[sample_indices])
        if agent == 2:
            sample_indices = np.random.choice(range(len(self.a2.validation.images)), count, replace=False)
            return Data(self.a2.validation.images[sample_indices], self.a2.validation.labels[sample_indices])


class MTData():
    '''

    '''

    def __init__(self):
        # gamma is the mixing factor
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    def mix_data(self, gamma):
        raw_mnist_train = Data(self.mnist.train.images, self.mnist.train.labels)
        raw_mnist_test = Data(self.mnist.test.images, self.mnist.test.labels)
        self.all_train = raw_mnist_train
        self.all_test = raw_mnist_test

        self.mnist1_train = self.sample_gamma(gamma, raw_mnist_train)
        self.mnist1_test = self.sample_gamma(gamma, raw_mnist_test)
        self.mnist2_train = self.sample_gamma(1.0 - gamma, raw_mnist_train)
        self.mnist2_test = self.sample_gamma(1.0 - gamma, raw_mnist_test)


        mnist2 = self.permute_mnist(self.mnist)
        mnist_permute_train = Data(mnist2.train.images, mnist2.train.labels)
        mnist_permute_test = Data(mnist2.test.images, mnist2.test.labels)
        self.all_train.add_data_obj(mnist_permute_train)
        self.all_test.add_data_obj(mnist_permute_test)

        self.mnist1_train.add_data_obj(self.sample_gamma(1.0 - gamma, mnist_permute_train))
        self.mnist1_test.add_data_obj(self.sample_gamma(1.0 - gamma, mnist_permute_test))
        self.mnist2_train.add_data_obj(self.sample_gamma(gamma, mnist_permute_train))
        self.mnist2_test.add_data_obj(self.sample_gamma(gamma, mnist_permute_test))


    def sample_gamma(self, gamma, data):
        sample_size = int(data.get_count()*gamma)
        indices = np.random.choice(np.arange(data.get_count()), sample_size, replace=True)
        return Data(data.images[indices], data.labels[indices])

    def permute_mnist(self, mnist):
        np.random.seed(10)
        perm_inds = list(range(mnist.train.images.shape[1]))
        np.random.shuffle(perm_inds)
        mnist2 = deepcopy(mnist)
        sets = ["train", "validation", "test"]
        for set_name in sets:
            this_set = getattr(mnist2, set_name)  # shallow copy
            this_set._images = np.transpose(np.array([this_set.images[:, c] for c in perm_inds]))
        return mnist2



def load_letters(self):
    emnist = spio.loadmat("matlab/emnist-digits.mat")
    x_train = emnist["dataset"][0][0][0][0][0][0]
    x_train = x_train.astype(np.float32)

    # load training labels
    y_train = emnist["dataset"][0][0][0][0][0][1]

    # load test dataset
    x_test = emnist["dataset"][0][0][1][0][0][0]
    x_test = x_test.astype(np.float32)

    # load test labels
    y_test = emnist["dataset"][0][0][1][0][0][1]

    x_train /= 255
    x_test /= 255

    train_labels = y_train
    test_labels = y_test

    y_train = self.get_one_hot(min_val=1, labels=train_labels)
    y_test = self.get_one_hot(min_val=1, labels=test_labels)
    letters_train = Data(x_train, y_train)
    letters_test = Data(x_test, y_test)
