import tensorflow as tf

class CNN:

    def __init__(self):

        self.sess = tf.Session()
        self.x = tf.placeholder(tf.float32, [None, 784], name='input')

        # Define loss and optimizer. The 11th output neuron is for "false"
        self.y_ = tf.placeholder(tf.float32, [None, 11], name="output")

        # Build the graph for the deep net
        y_conv, keep_prob = self.deepnn(self.x)

        with tf.name_scope('loss'):
            self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,
                                                                    logits=y_conv, name="output_loss")
        self.cross_entropy = tf.reduce_mean(self.cross_entropy, name="loss")

        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(self.y_, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
        self.accuracy = tf.reduce_mean(correct_prediction, name="accuracy")
        tf.add_to_collection('loss', self.accuracy)
        self.sess.run(tf.global_variables_initializer())

    def train(self, train_data, test_data, n=20000):
        with self.sess.as_default():
            for i in range(n):
                batch = train_data.next_batch(50)
                if i % 100 == 0:
                    train_accuracy = self.accuracy.eval(feed_dict={
                        self.x: batch.images, self.y_: batch.labels, self.keep_prob: 1.0})
                    print('step %d, training accuracy %g' % (i, train_accuracy))
                    print('test accuracy %g' % self.accuracy.eval(feed_dict={
                        self.x: test_data.images, self.y_: test_data.labels, self.keep_prob: 1.0}))
                self.train_step.run(feed_dict={self.x: batch.images, self.y_: batch.labels, self.keep_prob: 0.5})



    def deepnn(self, x):
        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32], name = "W")
            b_conv1 = bias_variable([32], name = "bias")
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool_2x2(h_conv1)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64], name = "W")
            b_conv2 = bias_variable([64], name = "b")
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool_2x2(h_conv2)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        with tf.name_scope('fc1'):
            W_fc1 = weight_variable([7 * 7 * 64, 1024], name = "w")
            b_fc1 = bias_variable([1024], name="b")

            h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of
        # features.
        with tf.name_scope('dropout'):
            self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
            self.h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob, name="to_feed")
            tf.add_to_collection('dropout', self.h_fc1_drop)
        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            self.W_fc2 = weight_variable([1024, 11], name= "final_W")
            self.b_fc2 = bias_variable([11], name= "final_b")

            y_conv = tf.matmul(self.h_fc1_drop, self.W_fc2) + self.b_fc2
        return y_conv, self.keep_prob

    def evaluate(self, train_data):
        return self.accuracy.eval(feed_dict={
            self.x: train_data.images, self.y_: train_data.labels, self.keep_prob: 1.0})

    def get_final_w(self):
        return self.W_fc2

    def get_final_b(self):
        return self.b_fc2



def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, name):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)