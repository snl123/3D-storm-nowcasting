import tensorflow as tf


def csi_score(input, label, downvalue, upvalue):
    ground_truth = tf.bitwise.bitwise_and(tf.cast(tf.greater_equal(label, downvalue), dtype=tf.int32),
                                          tf.cast(tf.less_equal(label, upvalue), dtype=tf.int32))
    prediction = tf.bitwise.bitwise_and(tf.cast(tf.greater_equal(input, downvalue), dtype=tf.int32),
                                        tf.cast(tf.less_equal(input, upvalue), dtype=tf.int32))
    x = tf.multiply(ground_truth, prediction)
    y = tf.multiply(ground_truth, tf.subtract(1, prediction))
    z = tf.multiply(prediction, tf.subtract(1, ground_truth))
    w = tf.multiply(tf.subtract(1, prediction), tf.subtract(1, ground_truth))
    x_ = tf.cast(tf.reduce_sum(x, axis=[2, 3, 4]), dtype=tf.float32)
    y_ = tf.cast(tf.reduce_sum(y, axis=[2, 3, 4]), dtype=tf.float32)
    z_ = tf.cast(tf.reduce_sum(z, axis=[2, 3, 4]), dtype=tf.float32)
    w_ = tf.cast(tf.reduce_sum(w, axis=[2, 3, 4]), dtype=tf.float32)
    CSI = tf.div(x_, x_ + y_ + z_)
    return CSI