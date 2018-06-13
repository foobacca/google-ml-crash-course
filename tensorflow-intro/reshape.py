import tensorflow as tf

# Write your code for Task 1 here.
with tf.Graph().as_default():
    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])
    re_a = tf.reshape(a, [2, 3])
    re_b = tf.reshape(b, [3, 1])

    matrix_multiply_result = tf.matmul(re_a, re_b)

    with tf.Session() as sess:
        print(matrix_multiply_result.eval())
