import tensorflow as tf
from RNN_cells.ConvLSTM3d_cell import ConvLSTM
import numpy as np

dtype = tf.float32

conv_activation_function = tf.nn.relu

convlstm_activation_function = tf.tanh
convlstm__initializer = tf.contrib.layers.xavier_initializer(seed=1997)
convlstm_feature_num = 64
convlstm_kernel_size = [3, 3]


def conv3d(scope, inputs, output_channels, kernel_shape, strides, activation_function=None, padding='same', is_training=False, initializer=tf.contrib.layers.xavier_initializer(seed=1997), reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        conv = tf.layers.conv3d(inputs, output_channels, kernel_shape, strides, padding=padding,
                                activation=activation_function, kernel_initializer=initializer)
    return conv

def up_conv3d(scope, inputs, output_channels, kernel_shape, strides, activation_function=None, padding='same', is_training=False, initializer=tf.contrib.layers.xavier_initializer(seed=1997), reuse=tf.AUTO_REUSE):
    with tf.variable_scope(scope, reuse=reuse):
        conv = tf.layers.conv3d_transpose(inputs, output_channels, kernel_shape, strides, padding=padding,
                                activation=activation_function, kernel_initializer=initializer)
    return conv

def generator(input, input_length, output_length, is_training=False, reuse=tf.AUTO_REUSE):
    batch_size = tf.shape(input)[0]

    channel = 16

    label = input[:, input_length:]

    one = tf.ones_like(label)
    MASK = tf.where(label < 15/80, one, label)
    MASK = tf.where((label >= 15 / 80)&(label < 35 / 80), one*3, MASK)
    MASK = tf.where((label >= 35 / 80)&(label < 45 / 80), one*8, MASK)
    MASK = tf.where(label >= 45 / 80, one*15, MASK)


    with tf.variable_scope("generator", reuse=reuse):

        input = tf.transpose(input[:, :input_length], [0, 4, 2, 3, 1])
        u3d_input = []
        for i in range(input_length):
            input_ = tf.expand_dims(input[:, :, :, :, i], axis=-1)
            spatial_features = spatial_extractor(input_, is_training, reuse)
            u3d_input.append(spatial_features)
        u3d_input = tf.transpose(u3d_input, [1, 0, 2, 3, 4, 5])


        input_shape = u3d_input.get_shape().as_list()[2:5]

        ConvLSTM1 = ConvLSTM('ConvLSTM1', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer, reuse=reuse)
        ConvLSTM1_outputs, ConvLSTM1_state = tf.nn.dynamic_rnn(ConvLSTM1, u3d_input, initial_state=ConvLSTM1.zero_state(), dtype=dtype, time_major=False)

        ConvLSTM2 = ConvLSTM('ConvLSTM2', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer, reuse=reuse)
        _, ConvLSTM2_state = tf.nn.dynamic_rnn(ConvLSTM2, ConvLSTM1_outputs, initial_state=ConvLSTM2.zero_state(), dtype=dtype, time_major=False)


        ConvLSTM3 = ConvLSTM('ConvLSTM3', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer, reuse=reuse)
        forecasting_input = tf.zeros(shape=[batch_size, output_length] + input_shape + [1], dtype=dtype)
        ConvLSTM3_outputs, _ = tf.nn.dynamic_rnn(ConvLSTM3, forecasting_input, initial_state=ConvLSTM2_state, dtype=dtype, time_major=False)

        ConvLSTM4 = ConvLSTM('ConvLSTM4', batch_size, input_shape, convlstm_feature_num, convlstm_kernel_size, convlstm_activation_function, convlstm__initializer, reuse=reuse)
        ConvLSTM4_outputs, _ = tf.nn.dynamic_rnn(ConvLSTM4, ConvLSTM3_outputs, initial_state=ConvLSTM1_state, dtype=dtype, time_major=False)

        out = []
        for i in range(output_length):
            out_ = decoder(ConvLSTM4_outputs[:,i], is_training, reuse)
            out.append(out_)

        out = tf.stack(out)
        out = tf.transpose(out, [1, 0, 3, 4, 2])
        loss1 = tf.abs(out - label)*MASK
        loss2 = tf.square(out - label)*MASK
        loss = tf.reduce_mean(loss1) + tf.reduce_mean(loss2)

        return [out, label, loss]


def spatial_extractor(input, is_training, reuse):

    with tf.variable_scope("spatial_extractor", reuse=reuse):
        batch_size = tf.shape(input)[0]
        altitude = input.get_shape().as_list()[1]
        input_shape = input.get_shape().as_list()[2:4]
        d1 = conv3d('d1', input, 32, [3, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(16,120,120,32)
        d2 = conv3d('d2', d1, 64, [3, 3, 3], [2, 2, 2], activation_function=conv_activation_function, is_training=is_training) #(8,60,60,64)
        d3 = conv3d('d3', d2, 64, [3, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(8,60,60,64)
        d4 = conv3d('d4', d3, 64, [3, 3, 3], [2, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(4,60,60,64)
    return d4

def decoder(input, is_training, reuse):
    with tf.variable_scope("decoder", reuse=reuse): #(b,4,60,60,64)
        u1 = up_conv3d('u1', input, 64, [3,3, 3], [2, 2, 2], activation_function=conv_activation_function, is_training=is_training) #(b,8,120,120,64)
        u2 = conv3d('u2', u1, 64, [1, 3, 3], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,8,120,120,64)
        u3 = up_conv3d('u3', u2, 64, [3,3, 3], [2, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,16,120,120,64)
        out = conv3d('out', u3, 1, [1, 1, 1], [1, 1, 1], activation_function=conv_activation_function, is_training=is_training) #(b,16,120,120,1)
        out = tf.squeeze(out, axis=-1)
    return out