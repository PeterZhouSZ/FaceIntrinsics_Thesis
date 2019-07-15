from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.eager as tfe
import imageio

import numpy as np
from data_loader import load_data_with_normals,plot_data_bis
from autoencoder_convolutional_V1 import FirstUVAutoencoder
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import envMapOrientation
import math

tf.enable_eager_execution()

import matplotlib.pyplot as plt

predict_sphere_envMap = (True,False)  # first argument : predict - second : sphere
load_pre_trained_model = False
learnable_envMap_size = (16, 32)
high_res_mode = True
adversarial_mode = True

show_every_n_steps = 5

def main():
    input_shape = (256, 256, 3)
    mIm = input_shape[0]
    nIm = input_shape[1]
    if (high_res_mode):
        mIm_res = 1024
        nIm_res = 1024
    else :
        mIm_res = mIm
        nIm_res = nIm
    # Load data
    data_folder_name = "Data04_isomaps_normals/"
    ratio_train = 0.75
    max_files = 5000
    (dataset_train, num_train) = load_data_with_normals(data_folder_name, input_shape,True)
    plot_data_bis(dataset_train, 0)
    plot_data_bis(dataset_train, 3)
    plot_data_bis(dataset_train, 2)

    epochs = 5000
    batch_size = 1
    num_batches = int(num_train / batch_size)

    envDir = "EnvMaps/"
    envName = envDir + "kitchen"
    envFile = envName + "_probe.hdr"
    envMap = imageio.imread(envName + '_probe.hdr', format='HDR-FI')
    (mEnv, nEnv, dEnv) = envMap.shape
    envMapTensor = tf.constant(envMap)
    if predict_sphere_envMap[0]:
        mEnv = learnable_envMap_size[0]
        nEnv = learnable_envMap_size[1]
    else :
        envmap_resize_rate = 0.04
        mEnv = int(envmap_resize_rate * mEnv)
        nEnv = int(envmap_resize_rate * nEnv)

    envMapTensor = tf.image.resize(envMapTensor, tf.constant([mEnv, nEnv]))
    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])
    print("(mEnv,nEnv) = {},{}".format(mEnv,nEnv) )
    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, predict_sphere_envMap[1])
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])
    if (predict_sphere_envMap[1]):
        envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
    else:
        envNormalization = tf.constant(float(mEnv * nEnv))


    envMapMask = envMapOrientation.getMaskSphereMap(mEnv,nEnv)
    #plt.imshow(envMapMask)
    #plt.show()
    envMapMaskTensor = tf.constant(envMapMask)

    checkpoint_write_dir = "model"#_10_samples_withEnvMap_OkResults/"
    checkpoint_load_dir = "model_10_samples_withEnvMap_OkResults/"
    checkpoint_write_prefix = checkpoint_write_dir + "autoencoder"

    #autoencoder_model = SecondUVAutoencoder(input_shape, envOrientationTensor,envMapTensor, envNormalization, predict_envMap)
    autoencoder_model = FirstUVAutoencoder(input_shape, envOrientationTensor,
                                           envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)

    lambda_normal_variance = 0.2

    optimizer = tf.train.AdamOptimizer()

    logdir = "./tb/"
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000)

    root = tf.train.Checkpoint(optimizer=optimizer,
                               model=autoencoder_model,
                               optimizer_step=tf.train.get_or_create_global_step())

    if (load_pre_trained_model):
        root.restore(tf.train.latest_checkpoint(checkpoint_load_dir))
        dataset_train = dataset_train.shuffle(num_train)
        x_train = dataset_train.batch(1)
        for (batch, (inputs, labels_appearance, labels_normals, masks)) in enumerate(x_train.take(4)):
            (albedos_preds, normals_preds, appearances_preds, envMap_preds) = autoencoder_model(inputs)
            plt.imshow(appearances_preds[0])
        plt.show()
        return

    num_itr = 0
    while (num_itr < epochs):
        dataset_train = dataset_train.shuffle(num_train)
        x_train = dataset_train.batch(batch_size)
        for (batch, (inputs, labels_appearance, labels_normals, masks)) in enumerate(x_train):
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                with tfe.GradientTape() as tape:
                    (albedos_preds,normals_preds,appearances_preds, envMap_preds) = autoencoder_model(inputs)
                    loss = autoencoder_model.loss(appearances_preds, normals_preds,
                                                  albedos_preds,labels_appearance, labels_normals,masks)

                grads = tape.gradient(loss, autoencoder_model.variables)
                optimizer.apply_gradients(zip(grads, autoencoder_model.variables),
                                          global_step=tf.train.get_or_create_global_step())

                if (num_itr % show_every_n_steps == 1):
                    if (high_res_mode):
                        mask_show = tf.image.resize(masks[0], tf.constant([mIm_res, nIm_res]))
                    else:
                        mask_show = masks[0]
                    tf.contrib.summary.scalar("loss", loss)
                    tf.contrib.summary.image("appearance input", tf.reshape(inputs[0], (1, mIm, nIm, 3)))
                    tf.contrib.summary.image("normal map (from 3DMM) ", tf.reshape(labels_normals[0], (1, mIm, nIm, 3)))
                    tf.contrib.summary.image("albedo prediction", tf.reshape(tf.multiply(mask_show, albedos_preds[0]),
                                                                             (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("normal map prediction",
                                             tf.reshape(tf.multiply(mask_show, normals_preds[0]),
                                                        (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("appearance result",
                                             tf.reshape(tf.multiply(mask_show, appearances_preds[0]),
                                                        (1, mIm_res, nIm_res, 3)))
                    if (predict_sphere_envMap[0]):
                        if (predict_sphere_envMap[1]):
                            envMap_show = tf.multiply(envMapMask, envMap_preds[0])
                        else:
                            envMap_show = envMap_preds[0]
                        tf.contrib.summary.image("envMap result", tf.reshape(envMap_show, (1, mEnv, nEnv, 3)))
                if batch % 2 == 0:
                    print("Iteration {}, batch: {} loss: {:.3f}".format(num_itr, batch, loss.numpy()))
        num_itr = num_itr + 1


    root.save(checkpoint_write_prefix)
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
