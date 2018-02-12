import numpy as np
from sklearn.model_selection import train_test_split
import os


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


    def __init__(self, output_digit = False, file_prefix="transformed_inputs/size_%d", digits=[]):
        assert len(digits) > 0
        self.train = Data()
        self.test = Data()
        for digit in digits:
            train_x = np.loadtxt("%s/digit_%d_train_acts.txt" % (file_prefix, digit))
            if output_digit:
                train_y = np.loadtxt("%s/digit_%d_train_labels.txt" % (file_prefix, digit))
            else:
                train_y = self.even_odd_labels(np.loadtxt("%s/digit_%d_train_labels.txt" % (file_prefix, digit)))
            self.train.add_data(train_x, train_y)
            test_x = np.loadtxt("%s/digit_%d_test_acts.txt" % (file_prefix, digit))
            if output_digit:
                test_y = np.loadtxt("%s/digit_%d_test_labels.txt" % (file_prefix, digit))
            else:
                test_y = self.even_odd_labels(np.loadtxt("%s/digit_%d_test_labels.txt" % (file_prefix, digit)))
            self.test.add_data(test_x, test_y)
            #print "Loaded Digit-%d. Train Length: %d\tTest Length: %d" % (digit, self.train.get_count(), self.test.get_count() )

    def even_odd_labels(self, digit_labels):
        labels = []
        for digit in digit_labels:
            if (np.argmax(digit)) % 2 == 0:
                labels.append([1, 0])
            else:
                labels.append([0, 1])
        return np.array(labels)













def get_new_labels(is_odd, count):
    if is_odd:
        return np.repeat([[0, 1]], count, axis=0)
    else:
        return np.repeat([[1, 0]], count, axis=0)

def convert_data(input_size, odds = [1,5], evens=[0,2]):
    g1_train, g1_test = load_transformed_digit_data(digit=odds[0], input_size=input_size)
    g2_train, g2_test = load_transformed_digit_data(digit=odds[1], input_size=input_size)

    g1_train.labels = get_new_labels(is_odd=True, count=g1_train.get_count())
    g1_test.labels = get_new_labels(is_odd=True, count=g1_test.get_count())

    g2_train.labels = get_new_labels(is_odd=True, count=g2_train.get_count())
    g2_test.labels = get_new_labels(is_odd=True, count=g2_test.get_count())


    d_train, d_test = load_transformed_digit_data(evens[0], input_size)
    g1_train.add_data(new_images=d_train.images, new_labels=get_new_labels(is_odd=False, count=d_train.get_count()))
    g1_test.add_data(new_images=d_test.images, new_labels=get_new_labels(is_odd=False, count=d_test.get_count()))

    d_train, d_test = load_transformed_digit_data(evens[1], input_size)
    g2_train.add_data(new_images=d_train.images, new_labels=get_new_labels(is_odd=False, count=d_train.get_count()))
    g2_test.add_data(new_images=d_test.images, new_labels=get_new_labels(is_odd=False, count=d_test.get_count()))




def get_even_odds(agent1, agent2):
    # collect digits for agent 1


    # collect digits for agent 2




    return g1_train, g1_test, g2_train, g2_test


def load_transformed_full_data():
    train_acts = np.loadtxt("transformed_inputs/train_acts.txt")
    test_acts = np.loadtxt("transformed_inputs/test_acts.txt")
    train_labels = np.loadtxt("transformed_inputs/train_labels.txt")
    test_labels = np.loadtxt("transformed_inputs/test_labels.txt")
    return Data(train_acts, train_labels), Data(test_acts, test_labels)

def load_transformed_digit_data(digit, input_size):
    train_acts = np.loadtxt("transformed_inputs/size_%d/digit_%d_train_acts.txt" % (input_size, digit))
    test_acts = np.loadtxt("transformed_inputs/size_%d/digit_%d_test_acts.txt" % (input_size, digit))
    train_labels = np.loadtxt("transformed_inputs/size_%d/digit_%d_train_labels.txt" % (input_size, digit))
    test_labels = np.loadtxt("transformed_inputs/size_%d/digit_%d_test_labels.txt" % (input_size, digit))
    return Data(train_acts, train_labels), Data(test_acts, test_labels)

