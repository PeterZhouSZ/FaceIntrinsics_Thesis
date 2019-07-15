## Adapted from Neural Face Editing : http://openaccess.thecvf.com/content_cvpr_2017/papers/Shu_Neural_Face_Editing_CVPR_2017_paper.pdf

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

bottle_neck_layer_size = 512
lambda_loss_normal = 1.

envMap_bottle_size = (2,4)

class FirstUVAutoencoder(tf.keras.Model):

    def __init__(self, input_shape, envOrientationTensor, envMapTensor, envNormalization, predict_envMap, high_res_mode):
        super(FirstUVAutoencoder, self).__init__()
        self._input_shape = [-1,input_shape[0],input_shape[1],input_shape[2]]
        self.mIm = input_shape[0]
        self.nIm = input_shape[1]
        self.envNormalization = envNormalization
        self.predict_envMap = predict_envMap[0]
        self.probe_sphere = predict_envMap[1]
        self.high_res_mode = high_res_mode

        self.conv1 = tf.layers.Conv2D(filters = 32, kernel_size= 3,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.conv2 = tf.layers.Conv2D(filters = 64, kernel_size= 3,
                                      padding='same',
                                      activation=tf.nn.relu)
        self.conv3 = tf.layers.Conv2D(filters=64, kernel_size=3,
                                      padding='same',
                                      activation=tf.nn.relu)

        self.fc1 = tf.layers.Dense(bottle_neck_layer_size, activation=tf.nn.relu)
        self.fc2 = tf.layers.Dense(bottle_neck_layer_size, activation=tf.nn.relu)
        self.fc3 = tf.layers.Dense(bottle_neck_layer_size, activation=tf.nn.relu)
        self.fc_env = tf.layers.Dense(bottle_neck_layer_size, activation=tf.nn.relu)

        layer_size = self.mIm / (2 * 2 * 2) * self.nIm / (2 * 2 * 2)
        self.fc4_normal = tf.layers.Dense(64 * layer_size, activation=tf.nn.relu)
        self.fc4_albedo = tf.layers.Dense(64 * layer_size, activation=tf.nn.relu)
        envMap_layer_size = envMap_bottle_size[0]*envMap_bottle_size[1]
        self.fc4_env =  tf.layers.Dense(64 * envMap_layer_size, activation=tf.nn.relu)


        self.deconv1_normal = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv2_normal = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv3_normal = tf.layers.Conv2D(filters=32, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv4_normal = tf.layers.Conv2D(filters=32, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv5_normal = tf.layers.Conv2D(filters=16, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.deconv1_albedo = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv2_albedo = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv3_albedo = tf.layers.Conv2D(filters=32, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv4_albedo = tf.layers.Conv2D(filters=32, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)
        self.deconv5_albedo = tf.layers.Conv2D(filters=16, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.deconv1_env = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.deconv2_env = tf.layers.Conv2D(filters=64, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.deconv3_env = tf.layers.Conv2D(filters=32, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.deconv_final_albedo = tf.layers.Conv2D(filters=3, kernel_size=3,
                                              padding='same',
                                              activation=tf.nn.sigmoid)
        self.deconv_final_normal = tf.layers.Conv2D(filters=3, kernel_size=3,
                                              padding='same',
                                              activation=tf.nn.sigmoid)
        self.deconv_final_env = tf.layers.Conv2D(filters=3, kernel_size=3,
                                               padding='same',
                                               activation=tf.nn.relu)

        self.max_pool2d = tf.layers.MaxPooling2D((2, 2), (2, 2), padding='same')
        self.up_sampl2d = tf.keras.layers.UpSampling2D(size = (2, 2))

        self.envOrientationTensor = envOrientationTensor
        self.envMapTensor = envMapTensor

    def render(self, normalTensor, albedoTensor):
        #normalTensorSign = normalTensor
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
        #normalTensorSign = normalTensor
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
        if (self.high_res_mode):
            pred_loss = tf.image.resize(pred, tf.constant([gt.shape[1],gt.shape[2]]))
        else :
            pred_loss = tf.reshape(pred, gt.shape)
        return tf.losses.mean_squared_error(labels=gt, predictions=pred_loss, weights=mask)

    def loss(self, pred_appearance, pred_normal, pred_albedo,gt_appearance, gt_normal, data_mask):
        loss_app = self.loss_l2(pred_appearance,gt_appearance, data_mask)
        loss_norm = self.loss_l2(pred_normal,gt_normal,data_mask)
        #loss_albedo = self.loss_l2(pred_albedo,gt_appearance,data_mask)
        #return loss_norm
        return loss_app  + lambda_loss_normal * loss_norm
        #return loss_app + lambda_loss_normal * loss_norm + lambda_loss_albedo * loss_albedo

    def call(self, x):
        x = tf.reshape(x, self._input_shape)
        x = self.max_pool2d(self.conv1(x))
        x = self.max_pool2d(self.conv2(x))
        x = self.max_pool2d(self.conv3(x))
        # x is size 32x32

        code = tf.layers.flatten(x)
        code = self.fc1(code) # code

        y = self.fc2(code)
        z = self.fc3(code)
        y = self.fc4_albedo(y)
        z = self.fc4_normal(z)

        y = tf.reshape(y,x.shape)
        y = self.up_sampl2d(self.deconv1_albedo(y))
        y = self.up_sampl2d(self.deconv2_albedo(y))
        y = self.up_sampl2d(self.deconv3_albedo(y))
        if (self.high_res_mode):
            y = self.up_sampl2d(self.deconv4_albedo(y))
            #y = self.up_sampl2d(self.deconv5_albedo(y))
        albedo = self.deconv_final_albedo(y)

        z = tf.reshape(z, x.shape)
        z = self.up_sampl2d(self.deconv1_normal(z))
        z = self.up_sampl2d(self.deconv2_normal(z))
        z = self.up_sampl2d(self.deconv3_normal(z))
        if (self.high_res_mode):
            z = self.up_sampl2d(self.deconv4_normal(z))
            #z = self.up_sampl2d(self.deconv5_normal(z))
        normal = self.deconv_final_normal(z)


        if (self.predict_envMap):
            w = self.fc_env(code)
            w = self.fc4_env(w)
            w = tf.reshape(w, [x.shape[0], envMap_bottle_size[0], envMap_bottle_size[1], x.shape[3]])
            w = self.up_sampl2d(self.deconv1_env(w))
            w = self.up_sampl2d(self.deconv2_env(w))
            w = self.up_sampl2d(self.deconv3_env(w))
            env_pred = self.deconv_final_env(w)
            res = self.render_with_predicted_envMap(normal, albedo,
                                                    tf.reshape(env_pred, [env_pred.shape[0],
                                                                          env_pred.shape[1] *
                                                                          env_pred.shape[2], 3]))
            return (albedo, normal, res, env_pred)


        res = self.render(normal, albedo)

        return (albedo,normal,res,self.envMapTensor)