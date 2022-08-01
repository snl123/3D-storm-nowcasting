import logging
import tensorflow as tf


def setting_log(log_save_path):
    logger = logging.getLogger("log")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(log_save_path, mode='a')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def imgs_summaries(imgs_seq, name_scope, max_outputs):
    with tf.name_scope(name_scope):
        for i in range(imgs_seq.get_shape().as_list()[0]):
            img_sum = tf.summary.image(name_scope + '/img' + str(i + 1), imgs_seq[i], max_outputs=max_outputs,
                                       collections='imgs_summaries')
            tf.add_to_collection('imgs_summaries', img_sum)


def variable_summaries(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(var.name + '_mean', mean)
    tf.summary.scalar(var.name + '_stddev', stddev)
    tf.summary.scalar(var.name + '_max', tf.reduce_max(var))
    tf.summary.scalar(var.name + '_min', tf.reduce_min(var))