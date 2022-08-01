import tensorflow as tf


class ConvLSTM(tf.contrib.rnn.RNNCell):
    def __init__(self, scope, batch_size, shape, feature_num, kernel_size, activation_function=tf.tanh, initializer=tf.contrib.layers.xavier_initializer(), regularizer=None, collection=None, state_is_tuple=True, dtype=tf.float32, reuse=tf.AUTO_REUSE):
        self.scope = scope
        self.batch_size = batch_size
        self.shape = shape
        self.feature_num = feature_num
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        self.initializer = initializer
        self.regularizer = regularizer
        self.collection = collection
        self.state_is_tuple = state_is_tuple
        self.dtype_ = dtype
        self.reuse = reuse
        self.c_size = tf.TensorShape(self.shape + [self.feature_num])
        self.h_size = tf.TensorShape(self.shape + [self.feature_num])

    @property
    def state_size(self):
        return (tf.contrib.rnn.LSTMStateTuple(self.c_size, self.h_size)) if self.state_is_tuple else 2 * self.feature_num

    @property
    def output_size(self):
        return self.h_size

    # zero state
    def zero_state(self):
        c = tf.zeros(shape=[self.batch_size] + self.shape + [self.feature_num], dtype=self.dtype_)
        h = tf.zeros(shape=[self.batch_size] + self.shape + [self.feature_num], dtype=self.dtype_)
        state = tf.contrib.rnn.LSTMStateTuple(c, h)
        return state

    def conv3d(self, scope, inputs, output_channels, kernel_shape, strides, activation_function=None, padding='same',
               is_training=False, initializer=tf.contrib.layers.xavier_initializer(seed=1997), reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            conv = tf.layers.conv3d(inputs, output_channels, kernel_shape, strides, padding=padding,
                                    activation=activation_function, kernel_initializer=initializer)
        return conv

    def __call__(self, input, state, scope=None):
        if scope is not None:
            call_scope = scope
        else:
            call_scope = self.scope
        with tf.variable_scope(call_scope, reuse=self.reuse):
            if self.state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(value=state, num_or_size_splits=2, axis=-1)

            output_channel = self.feature_num * 4
            input = tf.concat([input, h], axis=-1)
            concat = self.conv3d('concat', input, output_channel, [2, 3, 3], [1, 1, 1], activation_function=None)
            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=-1)
            i = tf.sigmoid(i)
            f = tf.sigmoid(f)
            o = tf.sigmoid(o)
            new_c = tf.multiply(c, f) + tf.multiply(i, tf.tanh(j))
            new_h = tf.multiply(self.activation_function(new_c), o)

            if self.state_is_tuple:
                new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat([new_c, new_h], axis=-1)

            return new_h, new_state
