"""
In order to simplify the structure of the code, any experiment will rely on this code here.

"""
from tensorflow.examples.tutorials.mnist import input_data
from data import *


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
        indices = np.random.choice(range(len(self.images)), n)
        return Data(self.images[indices], self.labels[indices])

    def get_count(self):
        return len(self.images)

class Agent:

    def __init__(self, type, train_images, train_labels, test_images, test_labels):
        if type == "eo":
            self.train = Data(train_images, self.even_odd_labels(train_labels))
            self.test = Data(test_images, self.even_odd_labels(test_labels))
        elif type == "digit":
            self.train = Data(train_images, train_labels)
            self.test = Data(test_images, test_labels)

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

        self.a1 = Agent(type, a1_train_images, a1_train_labels, a1_test_images, a1_test_labels)
        self.a2 = Agent(type, a2_train_images, a2_train_labels, a2_test_images, a2_test_labels)
        self.all = Agent(type, mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels)

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


