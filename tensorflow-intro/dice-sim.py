import tensorflow as tf

g = tf.Graph()
with g.as_default():
    first = tf.Variable(tf.random_uniform([10], minval=1, maxval=7, dtype=tf.int32))
    second = tf.Variable(tf.random_uniform([10], minval=1, maxval=7, dtype=tf.int32))
    sums = tf.add(first, second)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        first_10 = sess.run(first)
        second_10 = sess.run(second)
        sums_10 = sess.run(sums)
        throws = tf.stack([first_10, second_10, sums_10], axis=1)
        print(throws.eval())
