import tensorflow as tf


import tensorflow as tf

class Mnist2nn:

    def __init__(self, regularize = False):
        # Create the model
        self.sess = tf.Session()
        in_dim = int(784)  # 784 for MNIST
        out_dim = int(10)  # 10 for MNIST

        self.x = tf.placeholder(tf.float32, [None, 784])  # input placeholder
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        # simple 2-layer network
        W1 = self.weight_variable([in_dim, 50])
        b1 = self.bias_variable([50])

        W2 = self.weight_variable([50, out_dim])
        b2 = self.bias_variable([out_dim])

        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)  # hidden layer
        self.y = tf.matmul(h1, W2) + b2  # output layer

        self.var_list = [W1, b1, W2, b2]

        # vanilla single-task loss
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y))

        # performance metrics
        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.cross_entropy)
        self.sess.run(tf.global_variables_initializer())

    def train_model(self,train, good=True):
        for _ in range(4000):
            batch = train.next_batch(200)
            self.sess.run(self.train_step, feed_dict={self.x: batch.images, self.y_: batch.labels})

    def get_accuracy_cost(self, data):
        return self.sess.run([self.cross_entropy, self.accuracy], feed_dict={self.x: data.images,
                                                       self.y_: data.labels})

    def get_vars(self):
        return self.sess.run(self.var_list)

    def get_gradients(self, data):
        with self.sess.as_default():
            dc_dw, dc_db = tf.gradients(self.loss, [self.W, self.b], )
        a,b = self.sess.run([dc_dw, dc_db], feed_dict={self.x: data.images,
                                                       self.y_: data.labels})
        return a,b


    # variable initialization functions
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
