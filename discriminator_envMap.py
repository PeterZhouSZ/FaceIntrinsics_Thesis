
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

class FirstEnvMapDiscriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super(FirstEnvMapDiscriminator, self).__init__()

        self._input_shape = [-1,input_shape[0],input_shape[1],input_shape[2]]

        # self.conv0 = tf.layers.Conv2D(filters=16, kernel_size=3,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu)
        # self.conv1 = tf.layers.Conv2D(filters=32, kernel_size=3,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer)
        # self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=3,
        #                             padding='same',
        #                              activation=tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer)
        # self.conv3 = tf.layers.Conv2D(filters=128, kernel_size=3,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer)
        # self.conv4 = tf.layers.Conv2D(filters=256, kernel_size=3,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu)
        # self.conv5 = tf.layers.Conv2D(filters=256, kernel_size=3,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu)
        #
        # self.conv6 = tf.layers.Conv2D(filters=512, kernel_size=4, strides = 4,
        #                               padding='same',
        #                               activation=tf.nn.leaky_relu)

        self.conv1 = tf.layers.Conv2D(filters=32, kernel_size=1,
                                      padding='same', #strides = 2,
                                      activation=tf.nn.leaky_relu, kernel_initializer = tf.random_normal_initializer(stddev=0.02))
        self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=3,
                                      padding='same', #strides = 2,
                                      activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.conv3 = tf.layers.Conv2D(filters=128, kernel_size=3,
                                      padding='same', #strides = 2,
                                      activation=tf.nn.leaky_relu, kernel_initializer=tf.random_normal_initializer(stddev=0.02))


        self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')

        #self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.fc1 = tf.layers.Dense(1, activation=tf.keras.activations.sigmoid, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def call(self, x):
        #x = tf.reshape(x, self._input_shape)
        #x = self.max_pool2d(self.conv0(x))
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = self.max_pool2d(self.conv3(x))
        #x = self.batchnorm1(x)
        #x = self.max_pool2d(self.conv5(x))
        #x = self.conv6(x)
        x = tf.layers.flatten(x)
        output = self.fc1(x)
        return output





