import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import operator
from termcolor import colored
from tqdm import tqdm


class PartialModel:

    def __init__(self, input_size, window_width, output_size=10):
        self.sess = tf.Session()
        self.out_dim = output_size
        self.input_size = input_size
        self.window_width = window_width


        # Create training_model
        self.x = tf.placeholder(tf.float32, [None, self.input_size])  # input placeholder
        self.y_ = tf.placeholder(tf.float32, [None, self.out_dim])

        self.W = weight_variable([self.input_size, self.out_dim])
        self.b = bias_variable([self.out_dim])

        self.y = tf.matmul(self.x, self.W) + self.b  # output layer
        self.cross_entropy = tf.reduce_mean\
            (tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        # Create weight model
        self.w_x = tf.placeholder(tf.float32, [None, self.input_size])
        self.w_y_ = tf.placeholder(tf.float32, [None, self.out_dim])

        self.w_W = tf.placeholder(tf.float32, [self.input_size, self.out_dim])
        self.w_b = tf.placeholder(tf.float32, [self.out_dim])

        self.w_y = tf.matmul(self.w_x, self.w_W) + self.w_b  # output layer
        self.w_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.w_y_, logits=self.w_y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())

    '''
       Trains the model on the provided train data. 
    '''
    def train_model(self, train, test, iters=2000, verbose = False):
        loss = 0.0
        for iter in range(iters):
            batch = train.next_batch(200)
            self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_: batch.labels})
            if iter % 100 == 0:
                accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: test.images, self.y_: test.labels})
                verbose_print("Iters: %d, Loss: %f, Accuracy: %f" % (iter, loss, accuracy), verbose, True)
        accuracy, loss = self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: test.images, self.y_: test.labels})
        #print "Iters: %d, Loss: %f, Accuracy: %f" %(iter, loss, accuracy)
        return accuracy, loss

    def get_predictions(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data.images})

    def evaluate(self, data):
        return self.sess.run([self.accuracy, self.cross_entropy], feed_dict={self.x: data.images, self.y_: data.labels})

    def get_w(self):
        return self.sess.run(self.W)

    def get_b(self):
        return self.sess.run(self.b)

    def get_neuron_activations(self, neuron_index, data):
        return self.sess.run(self.y, feed_dict={self.x: data.images})[:, 0]

    def get_candidate_loss(self, new_w, new_b, test):
        return self.sess.run(self.w_cross_entropy, feed_dict={self.w_x: test.images, self.w_y_: test.labels, self.w_W: new_w, self.w_b: new_b})

    def vary_vars_w(self, test, verbose = False):
        default_w = self.sess.run(self.W)
        default_b = self.sess.run(self.b)
        self.ms_wy = np.zeros((self.input_size, self.out_dim, self.window_width))
        self.ms_wx = np.zeros((self.input_size, self.out_dim, self.window_width))
        for n in range(self.input_size):
            verbose_print("Neuron-%d" % n, verbose, True)
            for j in range(self.out_dim):
                new_w = default_w.copy()
                dw = default_w[n, j]
                vals_to_test = dw + np.arange(-0.5, 0.5, 1.0/self.window_width)
                #vals_to_test = np.append(-3, vals_to_test)
                #vals_to_test = np.append(vals_to_test, 3)
                verbose_print("V=%f: " % (dw), verbose, False)
                for i, new_val in enumerate(vals_to_test):
                    new_w[n, j] = new_val
                    loss = self.get_candidate_loss(new_w, default_b, test)
                    self.ms_wy[n, j, i] = loss
                    self.ms_wx[n, j, i] = round(new_val, 3)
                    if np.isclose(new_val,dw):
                        verbose_print(colored('%f' % loss, "blue"), verbose, False)
                    elif loss > 0.5:
                        verbose_print(colored('%f' % loss, "red"), verbose, False)
                    else:
                        verbose_print('%f' % loss, verbose, False)
                verbose_print("", verbose, True)

    def vary_vars_b(self, test, verbose = False):
        default_w = self.sess.run(self.W)
        default_b = self.sess.run(self.b)
        self.ms_by = np.zeros((self.out_dim, self.window_width))
        self.ms_bx = np.zeros((self.out_dim, self.window_width))
        for n in range(self.out_dim):
            verbose_print("Neuron-%d" % n, verbose, True)
            new_b = default_b.copy()
            db = default_b[n]
            vals_to_test = db + np.arange(-1.0, 1.0, 2.0/self.window_width)
            verbose_print("V=%f: " % (db), verbose, False)
            for i, new_val in enumerate(vals_to_test):
                new_b[n] = new_val
                loss = self.get_candidate_loss(default_w, new_b, test)
                self.ms_by[n, i] = loss
                self.ms_bx[n, i] = round(new_val, 3)
                if np.isclose(new_val,db):
                    verbose_print(colored('%.2E' % loss, "red"), verbose, False)
                else:
                    verbose_print('%.2E' % loss, verbose, False)
            verbose_print("", verbose, True)


    # Todo: Variable flattening
    def find_space(self, test, radius = 0.05, iters = 10, verbose=False):
        base_w = self.get_w()
        base_b = self.get_b()
        succed_count = 0.0
        for iter in range(iters):
            # Handle weights
            w_points = np.random.normal(loc=0.0, scale=1.0, size=base_w.shape)
            b_points = np.random.normal(loc=0.0, scale=1.0, size=base_b.shape)
            normalizer = 1.0 / np.sqrt(np.sum(np.square(w_points)) + np.sum(np.square(b_points)))
            new_w = base_w + w_points*radius*normalizer
            new_b = base_b + b_points * radius * normalizer
            loss = self.get_candidate_loss(new_w, new_b, test)

            if loss > 0.32:
                verbose_print(colored("Loss: %f" % (loss), "red"), verbose)
            else:
                succed_count += 1
                verbose_print(colored("Loss: %f" % (loss), "green"), verbose)

        return succed_count



    def set_w(self, new_w):
        self.sess.run(tf.assign(self.W, new_w))

    def set_b(self, new_b):
        self.sess.run(tf.assign(self.b, new_b))



def verbose_print(stmt, verbose, new_line = True):
    if verbose:
        if new_line:
            print stmt
        else:
            print stmt,
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def merge_vectors(L, X, window_width):

    offset = int((max(X[:, 0]) - min(X[:, 0])) / (2.0/window_width))
    new_x_shape = (X.shape[0], X.shape[1] + offset)
    new_x = np.zeros(new_x_shape)
    new_l = np.zeros(new_x_shape)
    #L = L / np.mean(L, axis=1).reshape((X.shape[0], 1))

    for i in range(new_x_shape[0]):
        x = X[i, :]
        l = L[i, :]
        f_off = int((x[0] - min(X[:, 0])) / (2.0/window_width))
        e_off = offset - f_off
        new_x[i, :] = np.lib.pad(x, (f_off, e_off), 'constant', constant_values=(x[0], x[-1]))
        new_l[i, :] = np.lib.pad(l, (f_off, e_off), 'constant', constant_values=(l[0], l[-1]))
    pred_losses = np.sum(new_l, axis=0)
    opt_x = np.mean(new_x[:, np.argmin(pred_losses)])

    return opt_x, np.min(pred_losses)


def merge_models_w(models, input_size, window_width, output_size=10):
    new_w = np.zeros((input_size, output_size))
    model_weights = [model.ms_wx for model in models]
    model_losses = [model.ms_wy for model in models]
    pred_loss_sum = 0.0
    for neuron in range(input_size):
        for w in range(output_size):
            weights = np.concatenate([[weight[neuron, w, :]] for weight in model_weights], axis=0)
            losses = np.concatenate([[losses[neuron, w, :]] for losses in model_losses], axis=0)
            w_opt, loss = merge_vectors(losses, weights, window_width)
            pred_loss_sum += loss
            new_w[neuron, w] = w_opt
    return new_w



def merge_models_b(models, window_width, output_size=10):
    new_b = np.zeros((output_size))
    model_biases = [model.ms_bx for model in models]
    model_losses = [model.ms_by for model in models]
    pred_loss = 0.0
    for neuron in range(output_size):
        biases = np.concatenate([[bias[neuron, :]] for bias in model_biases], axis=0)
        losses = np.concatenate([[losses[neuron, :]] for losses in model_losses], axis=0)
        b_opt, loss = merge_vectors(losses, biases, window_width=window_width)
        new_b[neuron] = b_opt
        pred_loss += loss
    return new_b

