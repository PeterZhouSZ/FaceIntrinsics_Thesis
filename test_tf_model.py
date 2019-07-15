from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import imageio

import numpy as np
from data_loader import load_data,plot_data
from autoencoder_convolutional_V1 import FirstUVAutoencoder
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import envMapOrientation
import math

tf.enable_eager_execution()

import matplotlib.pyplot as plt

predict_envMap = True

def main():
    input_shape = (256, 256, 3)
    mIm = input_shape[0]
    nIm = input_shape[1]
    # Load data
    data_folder_name = "Data01/" # "Data01/" # "Data02_isomaps/"
    ratio_train = 0.75
    max_files = 500000
    (dataset_train, num_train, dataset_test, num_test) = \
        load_data(data_folder_name, input_shape, ratio_train, max_files,False)
    plot_data(dataset_train, False)
    plot_data(dataset_train, True)

    epochs = 500
    batch_size = 10
    num_batches = int(num_train / batch_size)

    learnable_envMap_size = 16
    envDir = "EnvMaps/"
    envName = envDir + "kitchen"
    envFile = envName + "_probe.hdr"
    envMap = imageio.imread(envName + '_probe.hdr', format='HDR-FI')
    (mEnv, nEnv, dEnv) = envMap.shape
    envMapTensor = tf.constant(envMap)
    if predict_envMap:
        mEnv = learnable_envMap_size
        nEnv = learnable_envMap_size
    else :
        envmap_resize_rate = 0.04
        mEnv = int(envmap_resize_rate * mEnv)
        nEnv = int(envmap_resize_rate * nEnv)

    envMapTensor = tf.image.resize(envMapTensor, tf.constant([mEnv, nEnv]))
    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])
    print("(mEnv,nEnv) = {},{}".format(mEnv,nEnv) )
    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv)
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])
    envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)


    envMapMask = envMapOrientation.getMaskSphereMap(learnable_envMap_size,learnable_envMap_size)
    #plt.imshow(envMapMask)
    #plt.show()
    envMapMaskTensor = tf.constant(envMapMask)


    autoencoder_model = SecondUVAutoencoder(input_shape, envOrientationTensor, envMapTensor, envNormalization, envMapMaskTensor)
    lambda_normal_variance = 0.2

    def loss_variance(mask,y):
        (mean,variance) = tf.nn.moments(tf.math.multiply(mask,y),axes = [0,1,2])
        return lambda_normal_variance * tf.norm(variance)

    optimizer = tf.train.AdamOptimizer()

    logdir = "./tb/"
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000)

    num_itr = 0
    while (num_itr < epochs):
        dataset_train = dataset_train.shuffle(num_train)
        x_train = dataset_train.batch(batch_size)
        for (batch, (inputs, labels,masks)) in enumerate(x_train):
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                with tfe.GradientTape() as tape:
                    (albedos_preds,normals_preds,appearances_preds) = autoencoder_model(inputs)
                    loss = autoencoder_model.loss_l2_appearance(appearances_preds, labels,masks)

                grads = tape.gradient(loss, autoencoder_model.variables)
                optimizer.apply_gradients(zip(grads, autoencoder_model.variables),
                                          global_step=tf.train.get_or_create_global_step())
                tf.contrib.summary.scalar("loss", loss)
                tf.contrib.summary.image("appearance input", tf.reshape(inputs[0], (1, mIm, nIm, 3)))
                tf.contrib.summary.image("appearance label", tf.reshape(labels[0], (1, mIm, nIm, 3)))
                tf.contrib.summary.image("albedo prediction", tf.reshape(albedos_preds[0], (1, mIm, nIm, 3)))
                tf.contrib.summary.image("normal map prediction", tf.reshape(normals_preds[0], (1, mIm, nIm, 3)))
                tf.contrib.summary.image("appearance result", tf.reshape(appearances_preds[0], (1, mIm, nIm, 3)))
                if batch % 2 == 0:
                    print("Iteration {}, batch: {} loss: {:.3f}".format(num_itr, batch, loss.numpy()))
        num_itr = num_itr + 1

    return
    plt.figure(figsize=(20, 4))
    i = 0
    for (batch, (image, label, mask)) in enumerate(x_train):
        # Original
        subplot = plt.subplot(2, 10, i + 1)
        plt.imshow(image[0])
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)

        # Reconstruction
        subplot = plt.subplot(2, 10, i + 11)
        pred = autoencoder_model(image)
        pred = tf.reshape(pred, image.shape)
        plt.imshow(pred[0])
        subplot.get_xaxis().set_visible(False)
        subplot.get_yaxis().set_visible(False)
        i = i +1
        if (i>=10): break
    plt.show()

    return

if __name__ == "__main__":
    main()
