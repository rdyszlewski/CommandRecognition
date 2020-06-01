import tensorflow as tf
from model_config import ModelConfig


class Model:

    @staticmethod
    def create(configuration, input, reuse=False, is_training=True):
        input_size = configuration.data().input_size()
        num_classes = configuration.settings().num_classes()
        dropout = configuration.settings().dropout()

        conf = ModelConfig()

        with tf.variable_scope('ConvNet', reuse=reuse):
            # przeskalowanie do wymaganego rozmiaru
            x = tf.reshape(input, shape=[-1, input_size[0], input_size[1], 1])

            activation = Model.__get_activation(conf.conv_1_activation)
            # tworzenie warstw
            conv1 = tf.layers.conv2d(x, conf.conv_1_filters, conf.conv_1_size, activation=activation,
                                     padding=conf.conv_1_padding)
            pool1 = tf.layers.max_pooling2d(conv1, conf.pool_1_size, conf.pool_1_size)
            conv2 = tf.layers.conv2d(pool1, conf.conv_2_filters, conf.conv_2_size, activation=activation,
                                     padding=conf.conv_2_padding)
            pool2 = tf.layers.max_pooling2d(conv2, conf.conv_2_size, conf.conv_2_size)
            fc1 = tf.contrib.layers.flatten(pool2)
            fc1 = tf.layers.dense(fc1, conf.full_size)
            fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
            out = tf.layers.dense(fc1, num_classes)
            # out = tf.nn.dropout(out, dropout)
            # out = tf.nn.softmax(out)

        return out

    @staticmethod
    def __get_activation(activation):
        if activation == 'relu':
            return tf.nn.relu
