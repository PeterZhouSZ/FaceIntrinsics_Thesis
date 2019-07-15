## Adapted from Neural Face Editing : http://openaccess.thecvf.com/content_cvpr_2017/papers/Shu_Neural_Face_Editing_CVPR_2017_paper.pdf

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf


lambda_loss_normal = .1
lambda_loss_albedo = .1
class SecondUVAutoencoder(tf.keras.Model):
    # encoder:
    # C64 - C128 - C256 - C512 - C512 - C512 - C512 - C512
    # decoder:
    # CD512 - CD512 - CD512 - C512 - C256 - C128 - C64
    # U - Net decoder:
    # CD512 - CD1024 - CD1024 - C1024 - C1024 - C512 -C256 - C128


    def __init__(self, input_shape, envOrientationTensor, envMapTensor, envNormalization, predict_env_map):
        super(SecondUVAutoencoder, self).__init__()
        self._input_shape = [-1,input_shape[0],input_shape[1],input_shape[2]]
        self.mIm = input_shape[0]
        self.nIm = input_shape[1]
        self.envNormalization = envNormalization
        self.predict_env_map = predict_env_map
        self.conv1 = tf.layers.Conv2D(filters = 64, kernel_size= 4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)

        self.conv2 = tf.layers.Conv2D(filters = 128, kernel_size= 4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)

        self.conv3 = tf.layers.Conv2D(filters=256, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)
        self.conv4 = tf.layers.Conv2D(filters=512, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)
        self.conv5 = tf.layers.Conv2D(filters=512, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)
        self.conv6 = tf.layers.Conv2D(filters=512, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)
        self.conv7 = tf.layers.Conv2D(filters=512, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)
        self.conv8 = tf.layers.Conv2D(filters=512, kernel_size=4,
                                      padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                      activation=tf.nn.leaky_relu, strides=2)

        # CD512 - CD512 - CD512 - C512 - C256 - C128 - C64
        self.deconv1_normal = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout1_normal = tf.layers.Dropout()
        self.deconv2_normal = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout2_normal = tf.layers.Dropout()
        self.deconv3_normal = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout3_normal = tf.layers.Dropout()
        self.deconv4_normal = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv5_normal = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv6_normal = tf.layers.Conv2D(filters=256, kernel_size=4,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv7_normal = tf.layers.Conv2D(filters=128, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv8_normal = tf.layers.Conv2D(filters=64, kernel_size=4,
                                              padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                              activation=tf.nn.relu)
        self.deconv9_normal = tf.layers.Conv2D(filters=3, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.sigmoid)

        self.deconv1_albedo = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout1_albedo = tf.layers.Dropout()
        self.deconv2_albedo = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout2_albedo = tf.layers.Dropout()
        self.deconv3_albedo = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout3_albedo = tf.layers.Dropout()
        self.deconv4_albedo = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv5_albedo = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv6_albedo = tf.layers.Conv2D(filters=256, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv7_albedo = tf.layers.Conv2D(filters=128, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv8_albedo = tf.layers.Conv2D(filters=64, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv9_albedo = tf.layers.Conv2D(filters=3, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.sigmoid)

        self.deconv1_envMap = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout1_envMap = tf.layers.Dropout()
        self.deconv2_envMap = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.dropout2_envMap = tf.layers.Dropout()
        self.deconv3_envMap = tf.layers.Conv2D(filters=512, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv4_envMap = tf.layers.Conv2D(filters=256, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv5_envMap = tf.layers.Conv2D(filters=128, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)
        self.deconv6_envMap = tf.layers.Conv2D(filters=3, kernel_size=4,
                                               padding='same', kernel_initializer = tf.random_normal_initializer(stddev=0.02),
                                               activation=tf.nn.relu)

        self.up_sampl2d = tf.keras.layers.UpSampling2D(size = (2, 2))

        self.envOrientationTensor = envOrientationTensor
        self.envMapTensor = envMapTensor

    def render(self, normalTensor, albedoTensor):
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        cosineTensor = tf.matmul(normalTensorSign, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0, 1)
        shadedBrightness = tf.matmul(cosineTensor, self.envMapTensor)
        shadedBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(shadedBrightness,
                                                [n_shape[0], n_shape[1], n_shape[2], 3]))
        resTensor = tf.multiply(albedoTensor, shadedBrightness)
        return resTensor

    def render_with_predicted_envMap(self, normalTensor, albedoTensor, predicted_envMap):
        normalTensorSign = tf.scalar_mul(2., tf.add(-.5, normalTensor))
        n_shape = normalTensorSign.shape
        normalTensorSign = tf.reshape(normalTensorSign, [n_shape[0] * n_shape[1] * n_shape[2], 3])
        cosineTensor = tf.matmul(normalTensorSign, self.envOrientationTensor)
        cosineTensor = tf.clip_by_value(cosineTensor, 0, 1)
        cosineTensor = tf.reshape(cosineTensor, [n_shape[0],n_shape[1] * n_shape[2], cosineTensor.shape[1]])
        shadedBrightness = tf.matmul(cosineTensor, predicted_envMap)
        shadedBrightness = tf.scalar_mul(1. / self.envNormalization, tf.reshape(shadedBrightness,
                                                [n_shape[0], n_shape[1], n_shape[2], 3]))
        resTensor = tf.multiply(albedoTensor, shadedBrightness)
        return resTensor


    def loss_l2(self, pred, gt, mask):
        pred = tf.reshape(pred, gt.shape)
        return tf.losses.mean_squared_error(labels=gt, predictions=pred, weights=mask)

    def loss(self, pred_appearance, pred_normal, pred_albedo,gt_appearance, gt_normal, data_mask):
        loss_app = self.loss_l2(pred_appearance,gt_appearance, data_mask)
        loss_norm = self.loss_l2(pred_normal,gt_normal,data_mask)
        loss_albedo = self.loss_l2(pred_albedo,gt_appearance,data_mask)
        return loss_app
        #return loss_app + lambda_loss_normal * loss_norm + lambda_loss_albedo * loss_albedo


    def call(self, x):
        x = tf.reshape(x, self._input_shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        #x = self.conv7(x)
        #x = self.conv8(x)

        #y = self.up_sampl2d(self.deconv1_albedo(x))
        #y = self.dropout1_albedo(y)
        #y = self.up_sampl2d(self.deconv2_albedo(y))
        #y = self.dropout2_albedo(y)

        y = self.up_sampl2d(self.deconv3_albedo(x))
        y = self.dropout3_albedo(y)
        y = self.up_sampl2d(self.deconv4_albedo(y))
        y = self.up_sampl2d(self.deconv5_albedo(y))
        y = self.up_sampl2d(self.deconv6_albedo(y))
        y = self.up_sampl2d(self.deconv7_albedo(y))
        y = self.up_sampl2d(self.deconv8_albedo(y))
        albedo = self.deconv9_albedo(y)

        # z = self.up_sampl2d(self.deconv1_normal(x))
        # z = self.dropout1_normal(z)
        # z = self.up_sampl2d(self.deconv2_normal(z))
        # z = self.dropout3_normal(z)

        z = self.up_sampl2d(self.deconv3_normal(x))
        z = self.dropout3_normal(z)
        z = self.up_sampl2d(self.deconv4_normal(z))
        z = self.up_sampl2d(self.deconv5_normal(z))
        z = self.up_sampl2d(self.deconv6_normal(z))
        z = self.up_sampl2d(self.deconv7_normal(z))
        z = self.up_sampl2d(self.deconv8_normal(z))
        normal = self.deconv9_normal(z)

        if (self.predict_env_map):
            w = self.up_sampl2d(self.deconv1_envMap(x))
            w = self.dropout1_envMap(w)
            w = self.up_sampl2d(self.deconv2_envMap(w))
            w = self.dropout2_envMap(w)
            w = self.up_sampl2d(self.deconv3_envMap(w))
            w = self.up_sampl2d(self.deconv4_envMap(w))
            w = self.up_sampl2d(self.deconv5_envMap(w))
            envMapTensor_pred = self.deconv6_envMap(w)
            #envMapTensor_pred = tf.multiply(self.envMapMask,envMapTensor_pred)
            #envMapTensor_pred = tf.reshape(envMapTensor_pred, [envMapTensor_pred.shape[0], envMapTensor_pred.shape[1] * envMapTensor_pred.shape[2],3])
            res = self.render_with_predicted_envMap(normal, albedo,
                                                    tf.reshape(envMapTensor_pred, [envMapTensor_pred.shape[0], envMapTensor_pred.shape[1] * envMapTensor_pred.shape[2],3]))
            return (albedo, normal, res, envMapTensor_pred)
        res = self.render(normal, albedo)
        return (albedo,normal,res, self.envMapTensor)