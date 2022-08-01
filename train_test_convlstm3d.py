__author__ = 'Nengli Sun'

import os
import math
import configparser
import numpy as np
import shutil
import tensorflow as tf
from tfdeterminism import patch
patch()
import random
from datetime import datetime
from utils import log,  evaluation_functions
from model import ConvLSTM3d_model
import warnings
warnings.filterwarnings("ignore")

temporal_weights = np.arange(1,13)

def delete_file(path):
    jpg_files = os.listdir(path)
    for f in jpg_files:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

conf = configparser.ConfigParser()
conf.read("./configs.ini")

training_dataset_path = 'D:/gridrad/train8/'
validation_dataset_path = 'D:/gridrad/val8/'
test_dataset_path = 'D:/gridrad/test8/'

train_data_num = len(os.listdir('D:/gridrad/train7/'))
valid_data_num = len(os.listdir('D:/gridrad/val7/'))
test_data_num = len(os.listdir('D:/gridrad/test7/'))

running_log_path = conf.get('logfile_paths', 'running_log_path')

model_saving_path = conf.get('saver_configs', 'model_saving_path')
model_name = conf.get('saver_configs', 'model_name')
model_keep_num = int(conf.get('saver_configs', 'model_keep_num'))

generator_ini_learning_rate = float(conf.get('training_parameters', 'generator_ini_learning_rate'))
img_height = int(conf.get('training_parameters', 'img_height'))
img_width = int(conf.get('training_parameters', 'img_width'))
img_size = (img_height,img_width)
input_length = int(conf.get('training_parameters', 'input_length'))
output_length = int(conf.get('training_parameters', 'output_length'))
batch_size = int(conf.get('training_parameters', 'batch_size'))
val_batch_size = int(conf.get('training_parameters', 'val_batch_size'))
test_batch_size = int(conf.get('training_parameters', 'test_batch_size'))
training_steps = int(conf.get('training_parameters', 'training_steps'))
imgs_sum_steps = int(conf.get('training_parameters', 'imgs_sum_steps'))

patience = 20

def mkdir(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        return True
    else:
        return False

def seed_tensorflow(seed=1997):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


img_shape = (20,120,120,16)
def _parse_function(example_proto):
    features = {"img_raw": tf.FixedLenFeature((), tf.string),
              "name": tf.FixedLenFeature((), tf.string)}
    parsed_features = tf.parse_single_example(example_proto, features)
    img_str = parsed_features["img_raw"]
    name_str = parsed_features["name"]
    img = tf.decode_raw(img_str,tf.float16)
    img = tf.reshape(img,img_shape)
    img = tf.cast(img, tf.float32)/80
    name = tf.decode_raw(name_str,tf.int64)
    return img, name

def get_dataset_from_tfrecords(tfrecords_pattern,is_train_dataset,batch_size=1, threads=18, shuffle_buffer_size=1, cycle_length=1):

    if is_train_dataset:
        files = tf.data.Dataset.list_files(tfrecords_pattern, shuffle=True)
    else:
        files = tf.data.Dataset.list_files(tfrecords_pattern)

    dataset = files.interleave(map_func=tf.data.TFRecordDataset, cycle_length=cycle_length)
    dataset = dataset.map(_parse_function, num_parallel_calls=threads)

    if is_train_dataset:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size).batch(batch_size, drop_remainder=True).repeat(100).prefetch(buffer_size=batch_size)
    else:
        dataset = dataset.batch(batch_size, drop_remainder=False).repeat(100)
    return dataset


def main(argv=None):
    seed_tensorflow(seed=1997)
    update_generator = True
    if update_generator:
        delete_file(model_saving_path)
    if (os.path.isfile(running_log_path)) and update_generator:
        os.remove(running_log_path)
    with tf.device('/cpu:0'):
        logger = log.setting_log(running_log_path)

        tra_dataset = get_dataset_from_tfrecords(training_dataset_path + '*.tfrecords', is_train_dataset = True, batch_size=batch_size, shuffle_buffer_size=500)
        val_dataset = get_dataset_from_tfrecords(validation_dataset_path + '*.tfrecords', is_train_dataset = False, batch_size=val_batch_size, shuffle_buffer_size=500)
        test_dataset = get_dataset_from_tfrecords(test_dataset_path + '*.tfrecords', is_train_dataset = False, batch_size=test_batch_size, shuffle_buffer_size=500)
        # create the iterator

        tra_iterator = tra_dataset.make_one_shot_iterator()
        if update_generator:
            val_iterator = val_dataset.make_one_shot_iterator()
        else:
            val_iterator = test_dataset.make_one_shot_iterator()

        input, _ = tra_iterator.get_next()
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % 0) as scope:
                with tf.name_scope("tower_%d" % 0):

                    _, _, loss = ConvLSTM3d_model.generator(input, input_length, output_length, is_training=True)
                    with tf.name_scope('train_op'):
                        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                        with tf.control_dependencies(update_ops):
                            train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)
                    tf.get_variable_scope().reuse_variables()

        val_input, _ = val_iterator.get_next()
        with tf.variable_scope(tf.get_variable_scope()):
            with tf.device('/gpu:%d' % 0) as scope:
                with tf.name_scope("tower_%d" % 0):
                    prediction, label, valloss = ConvLSTM3d_model.generator(val_input, input_length, output_length, is_training=False)

        val_loss_tp = tf.placeholder(tf.float32)
        val_loss_sum = tf.summary.scalar('val_loss', val_loss_tp, collections='evalutaion_indexs')
        tf.add_to_collection('evalutaion_indexs', val_loss_sum)

        with tf.name_scope('evaluation_indexs'):
            CSI45 = evaluation_functions.csi_score(prediction, label, downvalue=45 / 80, upvalue=1)
            CSI = evaluation_functions.csi_score(prediction, label, downvalue=35 / 80, upvalue=1)

        train_saver = tf.train.Saver(max_to_keep=model_keep_num)
        validation_sum_steps = math.floor(train_data_num / batch_size)

        with tf.Session() as sess:
            logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " begin running -------------------")
            sess.run(tf.global_variables_initializer())
            logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " loading pretrain models -------------------")
            pre_tra_steps = 0
            model = tf.train.get_checkpoint_state(model_saving_path)
            if model and model.model_checkpoint_path:
                train_saver.restore(sess, model.model_checkpoint_path)
                pre_tra_steps = int(model.model_checkpoint_path.split('ckpt-')[1])
                logger.info(
                    "Found pretrain models, reading..... " + model.model_checkpoint_path + "   pretrain steps: " + str(
                        pre_tra_steps))
            else:
                logger.info("Not found pretrain models, restart training..... ")
            logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " training parameters -------------------")
            logger.info("train_data_num: " + str(train_data_num))
            logger.info("valid_data_num: " + str(valid_data_num))
            logger.info("test_data_num: " + str(test_data_num))
            logger.info("img_height: " + str(img_height))
            logger.info("img_width: " + str(img_width))
            logger.info("input_length: " + str(input_length))
            logger.info("output_length: " + str(output_length))
            logger.info("training_steps: " + str(training_steps))
            logger.info("imgs_sum_steps: " + str(imgs_sum_steps))
            logger.info("validation_sum_steps: " + str(validation_sum_steps))
            logger.info("batch_size: " + str(batch_size))
            logger.info("val_batch_size: " + str(val_batch_size))
            logger.info("test_batch_size: " + str(test_batch_size))
            logger.info("optimizer: " + str(tf.train.AdamOptimizer))
            logger.info("generator_ini_learning_rate: " + str(generator_ini_learning_rate))
            logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " model parameters num -------------------")
            logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " start training -------------------")

            count = 0
            best = -1
            model_step = pre_tra_steps
            if update_generator :

                for step in range(training_steps):

                    if update_generator and ((step + 1) % 1 == 0):
                        sess.run(train_op)
                    if (step + 1 + pre_tra_steps) % imgs_sum_steps == 0:
                        gen_train_loss = sess.run(loss)
                        logger.info(datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " training steps: %d, training loss: %g" % (step + 1 + pre_tra_steps, gen_train_loss))

                    if (step + 1 + pre_tra_steps) % validation_sum_steps == 0:

                        val_loss = 0
                        csi = []

                        batch_val_step = math.ceil(valid_data_num / val_batch_size) - 1

                        for i in range(batch_val_step):
                            val_loss_, csi_ = sess.run([valloss, CSI])
                            val_loss += val_loss_ * val_batch_size
                            csi.append(csi_)

                        val_loss_, csi_ = sess.run([valloss, CSI])
                        val_loss += val_loss_ * (valid_data_num-batch_val_step*val_batch_size)
                        csi.append(csi_)

                        csi = np.concatenate(csi, axis=0)
                        csi = np.nanmean(csi,axis=0)

                        val_loss = val_loss / valid_data_num

                        wcsi = np.sum(temporal_weights * csi) / np.sum(temporal_weights)
                        csi = list(csi)
                        logger.info(csi)
                        csi = np.nanmean(csi)
                        logger.info("------------------- Training steps: %d, validation loss: %g, CSI: %g, wCSI: %g" % (step + 1 + pre_tra_steps, val_loss, csi, wcsi) + " -------------------")

                        if wcsi <= best:
                            count += 1
                            logger.info('eval wcsi is not improved for {} epoch'.format(count) + '\n')
                            train_saver.save(sess, os.path.join(model_saving_path, model_name), step + 1 + pre_tra_steps)
                        else:
                            count = 0
                            logger.info('eval valpred_loss is improved from {:.5f} to {:.5f}, saving model'.format(best, wcsi) + '\n')
                            train_saver.save(sess, os.path.join(model_saving_path, model_name), step + 1 + pre_tra_steps)
                            best = wcsi

                        if count == patience:
                            print('early stopping reached, best score is {:5f}'.format(best))
                            logger.info('early stopping reached, best score is {:5f}'.format(best) + '\n')
                            model_step = step + 1 + pre_tra_steps
                            break

            if not update_generator:

                test_loss = 0
                csi45 = []
                csi = []

                batch_test_step = math.ceil(test_data_num / test_batch_size) - 1

                for i in range(batch_test_step):
                    test_loss_, csi45_, csi_, prediction_imgs_, label_imgs_ = sess.run([valloss, CSI45, CSI, prediction, label])

                    test_loss += test_loss_ * test_batch_size
                    csi45.append(csi45_)
                    csi.append(csi_)

                test_loss_, csi45_, csi_, prediction_imgs_, label_imgs_ = sess.run([valloss, CSI45, CSI, prediction, label])
                test_loss += test_loss_ * (test_data_num - batch_test_step * test_batch_size)
                csi45.append(csi45_)
                csi.append(csi_)

                csi45 = np.concatenate(csi45, axis=0)
                csi = np.concatenate(csi, axis=0)

                csi45 = np.nanmean(csi45, axis=0)
                csi = np.nanmean(csi, axis=0)

                test_loss = test_loss / test_data_num

                wcsi45 = np.sum(temporal_weights * csi45) / np.sum(temporal_weights)
                wcsi = np.sum(temporal_weights * csi) / np.sum(temporal_weights)

                csi45 = list(csi45)
                csi = list(csi)

                logger.info(csi45)
                logger.info('--------------------------------------------------------------------------')
                logger.info(csi)

                csi45 = np.nanmean(csi45)
                csi = np.nanmean(csi)

                logger.info(
                    "------------------- Training steps: %d, test loss: %g, CSI45: %g, CSI: %g, wCSI45: %g, wCSI: %g" % (
                    model_step, test_loss, csi45, csi, wcsi45, wcsi) + " -------------------")

                logger.info("------------------- Time: " + datetime.now().strftime('%Y-%m-%d  %H:%M:%S') + " end testing -------------------" + '\n')



if __name__ == '__main__':
    tf.app.run()