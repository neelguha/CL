import tensorflow as tf


graph = tf.Graph()
with graph.as_default():
    variable = tf.Variable(42, name='foo')
    initialize = tf.global_variables_initializer()
    assign = variable.assign(13)


with tf.Session(graph=graph) as sess:
  sess.run(initialize)
  sess.run(assign)
  print(sess.run(variable))
# Output: 13

with tf.Session(graph=graph) as sess:
  sess.run(initialize)
  print(sess.run(variable))
# Output: 42