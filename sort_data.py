# Sorts MNIST data into agents.
from collections import defaultdict
import numpy as np
from numpy.random import shuffle
from numpy import array_split

NUM_AGENTS = 10

train_data = defaultdict(list)
test_data = defaultdict(list)
sorted_data = defaultdict(list)

with open("mnist-csv/mnist_train.csv") as f:
    for line in f:
        items = line.split(",")
        train_data[int(items[0])].append(items[1:])

with open("mnist-csv/mnist_test.csv") as f:
    for line in f:
        items = line.split(",")
        test_data[int(items[0])].append(items[1:])

for z in range(10):
    print "%d: %d/%d" % (z, len(train_data[z]), len(test_data[z]))


for i in range(10):
    for image in train_data[i]:
        sorted_data[i].append((i, image))

for i in range(NUM_AGENTS):
    output_file = open("mnist-sorted/digit-%d-train.csv" % i, "w")
    for label, image in sorted_data[i]:
        output_file.write("%s,%s" % (label, ','.join(image)))