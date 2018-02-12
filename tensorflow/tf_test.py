import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(suppress=True,linewidth=np.nan,threshold=np.nan)
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def test_weight_place_holder():
    sess = tf.Session()

    x = tf.placeholder(tf.float32, [None, 3])  # input placeholder
    y_ = tf.placeholder(tf.float32, [None, 1])

    W = tf.placeholder(tf.float32, [None, 3])
    b = bias_variable([1])

    y = tf.matmul(x, W,transpose_b=True) + b  # output layer

    # vanilla single-task loss
    cross_entropy = tf.reduce_mean(tf.square(y_ - y), axis=0)
    sess.run(tf.global_variables_initializer())
    X = np.random.rand(10, 3)
    y = np.dot(X,[1,2,3])
    y = y.reshape((10,1))
    w_test = np.array([1,2,3, 4, 5, 6]).reshape((2,3))
    print sess.run(cross_entropy, feed_dict={x:X, y_:y, W: w_test})

def test_alignment():
    X = np.concatenate((np.arange(-0.29, 0.71, 0.1), np.arange(-0.37,0.63, 0.1), np.arange(0,1.0, 0.1))).reshape([3,10])
    L = np.random.rand(3,10)

    offset = int((np.ceil(max(X[:,0]) - min(X[:, 0])) / 0.1))
    print X, X.shape
    print "Offset: ",offset

    new_x_shape = (X.shape[0], X.shape[1]+offset)
    new_x = np.zeros(new_x_shape)
    new_l = np.zeros(new_x_shape)
    print "loss:",L, np.mean(L, axis=1)
    L = L / np.mean(L, axis=1).reshape((3,1))
    print L
    print
    for i in range(new_x_shape[0]):
        x = X[i,:]
        l = L[i,:]
        f_off = int(np.ceil((x[0] - min(X[:,0])) / 0.1))
        e_off = offset - f_off
        print x
        print "%d: %d, %d" % (i, f_off, e_off)
        print
        new_x[i,:] = np.lib.pad(x, (f_off, e_off), 'constant', constant_values=(x[0],x[-1]))
        new_l[i,:] = np.lib.pad(l, (f_off, e_off), 'constant', constant_values=(l[0],l[-1]))
    print new_x
    #print new_l
    print
    #print np.sum(new_l, axis=0)
    #print np.argmin(np.sum(new_l, axis=0))
    #print np.mean(new_x[:,np.argmin(np.sum(new_l, axis=0))])



test_alignment()

