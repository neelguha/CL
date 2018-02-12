import argparse
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
from cnn import *
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# specify agent
# Load base model if it exists, otherwise train it.
# Draw random weights for final layer
# Calculate loss
# Sort loss
# Write loss to file


BIAS_SHAPE = (10)
WEIGHT_SHAPE = (1024, 10)


class LossData:

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.x = []
        self.losses = []
        self.accuracy = []

    def get_line(self, weight, bias, loss, accuracy):
        output = np.concatenate((np.array([loss]), np.array([accuracy]), bias, weight.flatten()))
        return ','.join([str(v) for v in output]) + "\n"

    def add_line(self, line, threshold):
        items = line.strip().split(",")
        loss = items[0]
        print items[1]
        if float(items[1]) > threshold:
            accuracy = np.array([1.0, 0.0]).reshape((1,2))
        else:
            accuracy = np.array([0.0, 1.0]).reshape((1,2))
        input = items[2:]
        self.losses.append(loss)
        self.accuracy.append(accuracy)
        self.x.append(input)

    def get_data(self):
        return (np.array(self.x), np.array(self.accuracy).reshape((len(self.accuracy),2)))


class Data:
    ''' Represents a dataset.'''
    def __init__(self, x, y):
        self.images = x
        self.labels = y

    def next_batch(self, n):
        if n > len(self.images):
            raise ValueError("Batch size exceeds data")
        indices = np.random.choice(range(len(self.images)), n)
        return Data(self.images[indices], self.labels[indices])


class LastLayer():
    def __init__(self):
        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 1024])

        self.y_ = tf.placeholder(tf.float32, [None, 11])

        self.W_fc2 = weight_variable([1024, 11],"last-weight")
        self.b_fc2 = bias_variable([11],"last-bias")

        y_conv = tf.matmul(self.x, self.W_fc2) + self.b_fc2

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y_conv)
        self.cross_entropy = tf.reduce_mean(self.cross_entropy, name="loss")
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction)

    def get_loss(self, new_w, new_b, partials, labels):
        self.sess.run(tf.assign(self.W_fc2, new_w))
        self.sess.run(tf.assign(self.b_fc2, new_b))
        loss, accuracy =  self.sess.run([self.cross_entropy, self.accuracy],feed_dict={self.x:partials, self.y_:labels})
        return (loss, accuracy)
# Since we only care about the last layer, there's no need to repeatedly feed the images through the entire network
def get_partial_outputs(model_name, train):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(model_name)
    saver.restore(sess, tf.train.latest_checkpoint('base_models/agent-%d/.' % FLAGS.agent_id))
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input:0")
    y = graph.get_tensor_by_name("output:0")
    w = graph.get_tensor_by_name("fc2/final_W:0")
    b = graph.get_tensor_by_name("fc2/final_b:0")
    dropout_prob = graph.get_tensor_by_name("dropout/dropout_prob:0")
    z = tf.get_collection('dropout')[0]
    return sess.run([z,w,b],feed_dict={x:train.images, y:train.labels, dropout_prob:1.0})


# Mask = control whether we mask the non-agent classes. For example, on agent-0, all non zero digits will appear as "Falses"
def get_data(agent_id, mask=True):
    labels = []
    images = []
    with open("%s/mnist-sorted/agent-%d-train.csv" % (os.getcwd(), agent_id), 'r') as f:
        print "Reading training data for Agent-%d" % agent_id
        for line in f:
            items = [int(x) for x in line.strip().split(",")]
            label = np.zeros(11)
            if mask:
                if items[0] == agent_id:
                    label[items[0]] = 1
                else:
                    label[10] = 1
            else:
                label[items[0]] = 1
            labels.append(label)
            images.append(items[1:])
    data_train, data_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.20, random_state=42)
    return Data(np.array(data_train), np.array(labels_train)), Data(np.array(data_test), np.array(labels_test))

def create_model(train, test):
    agent_model = CNN()
    agent_model.train(train, test, 300)
    saver = tf.train.Saver()
    saver.save(agent_model.sess, 'base_models/agent-%d/test_model' % FLAGS.agent_id, global_step=1000)
    print "Writing model."

# Return a new parameter with some amount of gaussian perturbation
def new_parameter(param, var=0.1):
    return param + np.random.normal(0, 0.1,param.shape)

# Because neurons are interchangeable, we sort them in descending order by the row norm. This standardizes the weight space.
def sort_weights(weight):
    row_norms = np.amax(weight, axis=1)
    ordered_indices = np.argsort(row_norms)
    return weight#[ordered_indices]

def main(_):
    train, test = get_data(FLAGS.agent_id, mask=True)
    if FLAGS.restart:
        print "Training models from scratch."
        create_model(train, test)

    print "Reading models from '%s/'" % FLAGS.saved_model_dir
    partials,base_w,base_b = get_partial_outputs('base_models/agent-%d/test_model-1000.meta' % FLAGS.agent_id, test)
    m = LastLayer()
    output_file = open("loss-logs/agent-%d/losses.csv" % FLAGS.agent_id, "w")
    l = LossData(FLAGS.agent_id)
    print "Sampling %d points..." % FLAGS.num_points
    loss, accuracy = m.get_loss(base_w, base_b, partials, test.labels)
    output_file.write(l.get_line(base_w, base_b, loss, accuracy))
    print "Base loss: %f Base accuracy: %f" % (loss, accuracy)
    for i in range(FLAGS.num_points):
        new_w = sort_weights(new_parameter(base_w))
        new_b = new_parameter(base_b)
        loss, accuracy = m.get_loss(new_w,new_b,partials, test.labels)
        print  "(%d) Loss = %f Accuracy = %f" % (i, loss, accuracy)
        output_file.write(l.get_line(new_w, new_b, loss, accuracy))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='mnist',
                        help='Directory for storing input data')
    parser.add_argument('--saved_model_dir', type=str,
                        default='models',
                        help="Directory for saved models (lower layers)")
    parser.add_argument('--restart', dest='restart', action='store_true')
    parser.add_argument('--no-restart', dest='restart', action='store_false')
    parser.set_defaults(restart=False)
    parser.add_argument('--agent_id', type=int,
                        default=0,
                        help='Agent ID')
    parser.add_argument('--num_points', type=int,
                        default=100,
                        help='Number of loss points to sample')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

