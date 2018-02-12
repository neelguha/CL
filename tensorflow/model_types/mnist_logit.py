import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import operator

class MnistLogit:

    def __init__(self, num_neurons, w1 = [], w2 = [], b1 = [], b2 = []):
        self.sess = tf.Session()

        in_dim = int(784)  # 784 for MNIST
        out_dim = int(10)  # 10 for MNIST
        # Create the model
        self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
        self.y_ = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        # simple 2-layer network
        self.W1 = weight_variable([in_dim, num_neurons])
        self.b1 = bias_variable([num_neurons])

        self.W2 = weight_variable([num_neurons, out_dim])
        self.b2 = bias_variable([out_dim])

        self.h1 = tf.nn.relu(tf.matmul(self.x, self.W1) + self.b1)  # hidden layer
        self.h_fc1_drop = tf.nn.dropout(self.h1, self.keep_prob)
        self.y = tf.matmul(self.h_fc1_drop, self.W2) + self.b2  # output layer

        self.var_list = [self.W1, self.b1, self.W2, self.b2]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())
        if w1 != []:
            self.sess.run(tf.assign(self.W1, w1))
            self.sess.run(tf.assign(self.W2, w2))
            self.sess.run(tf.assign(self.b1, b1))
            self.sess.run(tf.assign(self.b2, b2))

    '''
       Trains the model on the provided train data. 
    '''
    def train_model(self, train, test, iters=2000, verbose = True):
        loss = 0.0
        for iter in range(iters):
            batch = train.next_batch(200)
            self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_: batch.labels, self.keep_prob: 0.5})
            if iter % 100 == 0 and verbose:
                accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: test.images, self.y_: test.labels, self.keep_prob: 1.0})
                print "Iters: %d, Loss: %f, Accuracy: %f" % (iter, loss, accuracy)

    def evaluate(self, data, verbose=True):
        accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: data.images, self.y_: data.labels, self.keep_prob: 1.0})
        if verbose:
            print "Loss: %f, Accuracy: %f" %(loss, accuracy)
        return loss, accuracy

    def get_predictions(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data.images})



    def get_neuron_activations(self, neuron_index, data):
        return self.sess.run(self.y, feed_dict={self.x: data.images})[:, 0]


    def swap_neuron(self, target_n, new_w, new_b, data):
        old_n_w = self.get_w1()
        old_n_b = self.get_b1()
        new_n_w = old_n_w.copy()
        new_n_b = old_n_b.copy()
        new_n_w[:,target_n] = new_w
        new_n_b[target_n] = new_b
        self.sess.run(tf.assign(self.W1, new_n_w))
        self.sess.run(tf.assign(self.b1, new_n_b))
        self.evaluate(data)
        self.sess.run(tf.assign(self.W1, old_n_w))
        self.sess.run(tf.assign(self.b1, old_n_b))




    def vary_vars(self, train, test):

        ss = np.zeros((784, 10))
        for image in train.images[:100]:
            i_t = tf.reshape(tf.convert_to_tensor(image, dtype=tf.float32), shape=(784, 1))
            ww = self.sess.run(i_t*self.W)
            ss += np.abs(ww)
        ss /= 100
        np.savetxt('scores.txt', np.array(ss), '%5.2f')

        # Convert to dictionary
        index_vals = {}
        for i in range(ss.shape[0]):
            for j in range(ss.shape[1]):
                index_vals[(i,j)] = ss[i,j]
        sorted_x = sorted(index_vals.items(), key=operator.itemgetter(1), reverse=True)


        base_accuracy, base_loss = self.evaluate(test)
        for i in range(100):
            index, val = sorted_x[i]
            default_w = self.sess.run(self.W)
            dw = default_w[index[0], index[1]]
            print "V-",index,",", dw, ": ",
            vals_to_test = dw + np.arange(-50, 50, 10)
            new_w = default_w.copy()
            for new_val in vals_to_test:
                new_w[index[0], index[1]] = new_val
                self.sess.run(tf.assign(self.W, new_w))
                acc, loss = self.evaluate(test)
                #print "\t", new_val, np.abs(acc-base_accuracy), np.abs(loss - base_loss)
                print np.abs(np.log(base_loss) - np.log(loss)),
            print
            self.sess.run(tf.assign(self.W, default_w))

    def merge_model(self, new_model):
        new_w = (new_model.sess.run(new_model.W) + self.sess.run(self.W))*0.5
        self.sess.run(tf.assign(self.W, new_w))

    def l1_activations(self, data):
        return self.sess.run(self.h1, feed_dict={self.x:data.images})

    def set_w1(self, w1):
        self.sess.run(tf.assign(self.W1, w1))

    def set_w2(self, w2):
        self.sess.run(tf.assign(self.W2, w2))

    def set_b1(self, b1):
        self.sess.run(tf.assign(self.b1, b1))

    def set_b2(self, b2):
        self.sess.run(tf.assign(self.b2, b2))

    def get_w1(self):
        return self.sess.run(self.W1)

    def get_w2(self):
        return self.sess.run(self.W2)

    def get_b1(self):
        return self.sess.run(self.b1)

    def get_b2(self):
        return self.sess.run(self.b2)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


