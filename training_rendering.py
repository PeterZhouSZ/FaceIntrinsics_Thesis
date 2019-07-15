from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from load_rgb_cv import load_rgb,convert_rgb_to_cv2,tonemap,save_tensor_cv2

import numpy as np
from data_loader import load_input_data_with_normals, \
    load_input_data_with_normals_and_replicate, load_ground_truth_data, \
    plot_data_bis,load_input_data_with_albedo_and_envmaps, plot_data_samples,plot_data_batches
from generator_V1 import FirstGenerator
from generator_V2 import SecondGenerator
from discriminator_V1 import FirstDiscriminator
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import random

import envMapOrientation
import math
import cv2
import sys

tf.enable_eager_execution()

sys.setrecursionlimit(100000)

import matplotlib.pyplot as plt

predict_sphere_envMap = (True,False)  # first argument : predict - second : sphere
load_pre_trained_model = False
learnable_envMap_size = (16, 32)
high_res_mode = False
adversarial_mode = True

show_every_n_steps = 8 # tensorboard
test_every_n_steps = 64
log_every_n_steps = 2
write_every_n_steps = 16

lambda_adv = .01

learning_rate_disc = 1e-4

gamma = tf.constant(2.2)
invGamma = tf.constant(1./2.2)

epochs = 10000
batch_size = 1

def main():
    input_shape = (256, 256, 3)
    shape_gt_adv = (512,512,3)
    mIm = input_shape[0]
    nIm = input_shape[1]
    if (high_res_mode):
        mIm_res = shape_gt_adv[0]
        nIm_res = shape_gt_adv[1]
    else :
        mIm_res = mIm
        nIm_res = nIm
    # Load data
    data_training_folder_name = "Synthesized_Training_1tris/"
    data_testing_folder_name = "Synthesized_Testing_1/"
    result_training_folder_name = data_training_folder_name + "Results/"
    (dataset_train_input, num_train_input) = load_input_data_with_albedo_and_envmaps(data_training_folder_name,
                                            learnable_envMap_size,
                                            input_shape, True)

    (dataset_test_input, num_test_input) = load_input_data_with_albedo_and_envmaps(data_testing_folder_name,
                                                                                     learnable_envMap_size,
                                                                                     input_shape, True)


    dataset_train_input = dataset_train_input.shuffle(num_train_input)
    x_train = dataset_train_input.batch(1)
    indices_plot = [0,2,3,4]#,5]
    plot_data_batches(x_train, indices_plot,4)




    num_batches = int(num_train_input / batch_size)

    #albedoMapGT = load_rgb("Chicago_albedo.png")
    #albedoMapGT = cv2.resize(albedoMapGT, (256,256))
    #albedoMapTensor = tf.constant(albedoMapGT)
    #albedoMapTensor = tf.reshape(albedoMapTensor, [1,256,256,3])
    envDir = "EnvMaps/"
    envName = envDir + "village.hdr"
    envMap = load_rgb(envName,-1)
    (mEnv, nEnv, dEnv) = envMap.shape
    envMap = cv2.resize(envMap,(32,16), interpolation = cv2.INTER_LINEAR)


    for (batch,(inputs, labels_appearance, labels_normals, masks, gt_albedo, labels_envmap)) in enumerate(dataset_train_input.take(1)):
        envMapTensor = labels_envmap
        envMapTensor = tf.reshape(envMapTensor, [1, 16, 32, 3])
        envMapTensorShow = tonemap(envMapTensor, gamma)
        plt.imshow(envMapTensorShow[0])
        plt.show()
    if predict_sphere_envMap[0]:
        mEnv = learnable_envMap_size[0]
        nEnv = learnable_envMap_size[1]
    else :
        envmap_resize_rate = 0.04
        mEnv = int(envmap_resize_rate * mEnv)
        nEnv = int(envmap_resize_rate * nEnv)

    print("(mEnv,nEnv) = {},{}".format(mEnv,nEnv) )
    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, predict_sphere_envMap[1])
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])
    if (predict_sphere_envMap[1]):
        envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
    else:
        envNormalization = tf.constant(float(mEnv * nEnv))

    checkpoint_write_dir = "model_saved"#_10_samples_withEnvMap_OkResults/"
    checkpoint_load_dir = "model_10_samples_withEnvMap_OkResults/"
    checkpoint_write_prefix = checkpoint_write_dir + "adversarial"

    #autoencoder_model = SecondUVAutoencoder(input_shape, envOrientationTensor,envMapTensor, envNormalization, predict_envMap)
    generator_model = SecondGenerator(input_shape, envOrientationTensor,
                                           envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)
    #discriminator_model = FirstDiscriminator()

    generator_optimizer = tf.train.AdamOptimizer()
    #discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate_disc)

    logdir = "./tb/"
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000)
    log_losses_ratio = 100.

    root = tf.train.Checkpoint(generator_model = generator_model)

    if (load_pre_trained_model):
        root.restore(tf.train.latest_checkpoint(checkpoint_load_dir))
        dataset_train_input = dataset_train_input.shuffle(num_train_input)
        x_train = dataset_train_input.batch(1)
        for (batch, (inputs, labels_appearance, labels_normals, masks)) in enumerate(x_train.take(4)):
            (albedos_preds, normals_preds, appearances_preds, envMap_preds) = generator_model(inputs)
            plt.imshow(appearances_preds[0])
        plt.show()
        return

    num_itr = 0
    training_step = -1
    while (num_itr < epochs):
        dataset_train_input = dataset_train_input.shuffle(num_train_input)
        x_train_input = dataset_train_input.batch(batch_size)
        x_test_input = dataset_test_input.shuffle(num_test_input).batch(1)
        for (batch, ((inputs, labels_appearance, labels_normals, masks, gt_albedo, labels_envmap), \
                     (inputs_test, labels_appearance_test, labels_normals_test, masks_test, gt_albedo_test, labels_envmap_test))) \
                in enumerate(zip(x_train_input,x_test_input)):

            #plt.imshow(ground_truth_images[0])
            #plt.show()
            perform_testing = False
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                with tf.GradientTape() as gen_tape:#, tf.GradientTape() as disc_tape:
                    (albedos_preds,normals_preds,appearances_preds, envMap_preds) = generator_model(inputs)

                    #fake_adv_output = discriminator_model(appearances_preds)
                    #real_adv_output = discriminator_model(ground_truth_images)

                    (loss_app,loss_norm,loss_env) = generator_model.loss_with_envmap(appearances_preds, normals_preds,
                                                  albedos_preds,labels_appearance, labels_normals,masks,
                                                                   envMap_preds, labels_envmap)

                    #gen_loss_adv = discriminator_model.generator_loss(fake_adv_output)

                    gen_loss = loss_app + loss_norm + loss_env # + lambda_adv * gen_loss_adv
                    #dis_loss = discriminator_model.discriminator_loss(real_adv_output, fake_adv_output)


                gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.variables)
                #gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator_model.variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.variables),
                                                    global_step=tf.train.get_or_create_global_step())
                # discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                #                                             discriminator_model.variables),
                #                                         global_step=tf.train.get_or_create_global_step())

                training_step += 1

                if (training_step % test_every_n_steps == 0):
                    perform_testing = True
                    (albedos_preds_test, normals_preds_test, appearances_preds_test, envMap_preds_test) = generator_model(inputs_test)
                    testing_loss = generator_model.loss_testing(appearances_preds_test,labels_appearance_test,masks_test)

                if (training_step % show_every_n_steps == 0):
                    tf.contrib.summary.scalar("generator total loss", gen_loss)
                    tf.contrib.summary.scalar("generator appearance loss", loss_app)
                    tf.contrib.summary.scalar("generator envmap loss", loss_env)
                    tf.contrib.summary.scalar("generator normal loss", loss_norm)
                    # tf.contrib.summary.scalar("adversarial loss (generator)", gen_loss_adv)
                    # tf.contrib.summary.scalar("adversarial loss (discriminator)", dis_loss)
                    tf.contrib.summary.image("appearance input (label)", tf.reshape(inputs[0], (1, mIm, nIm, 3)))
                    tf.contrib.summary.image("normal map (from 3DMM) ", tf.reshape(labels_normals[0], (1, mIm, nIm, 3)))
                    albedo_show = tf.pow(albedos_preds[0], invGamma)
                    tf.contrib.summary.image("albedo prediction", tf.reshape(albedo_show,
                                                                             (1, mIm_res, nIm_res, 3)))

                    albedoMapTensor = tf.reshape(gt_albedo[0], (1, mIm_res, nIm_res, 3))
                    tf.contrib.summary.image("albedo ground truth", albedoMapTensor)

                    tf.contrib.summary.image("normal map prediction",
                                             tf.reshape(normals_preds[0],
                                                        (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("appearance result",
                                             tf.reshape(appearances_preds[0],
                                                        (1, mIm_res, nIm_res, 3)))
                    if (predict_sphere_envMap[0]):
                        envMap_show = tonemap(envMap_preds[0],gamma)
                        tf.contrib.summary.image("envMap result", tf.reshape(envMap_show, (1, mEnv, nEnv, 3)))
                        envMapTensorShow = tf.reshape(tonemap(labels_envmap[0], gamma), (1, mEnv, nEnv, 3))
                        tf.contrib.summary.image("envMap ground truth", envMapTensorShow)

                    if (perform_testing):
                        tf.contrib.summary.scalar("Testing loss", testing_loss)
                        tf.contrib.summary.image("Testing appearance prediction", tf.reshape(appearances_preds_test[0],
                                                        (1, mIm_res, nIm_res, 3)))
                        tf.contrib.summary.image("Testing appearance label",
                                                tf.reshape(labels_appearance_test[0],
                                                            (1, mIm_res, nIm_res, 3)))
                        tf.contrib.summary.image("Testing normal map (from 3DMM) ",
                                                 tf.reshape(labels_normals_test[0], (1, mIm, nIm, 3)))
                        albedo_show = tf.pow(albedos_preds_test[0], invGamma)
                        tf.contrib.summary.image("Testing albedo prediction", tf.reshape(albedo_show,
                                                                                 (1, mIm_res, nIm_res, 3)))

                        albedoMapTensor = tf.reshape(gt_albedo_test[0], (1, mIm_res, nIm_res, 3))
                        tf.contrib.summary.image("Testing albedo ground truth", albedoMapTensor)

                        tf.contrib.summary.image("Testing normal map prediction",
                                                 tf.reshape(normals_preds_test[0],
                                                            (1, mIm_res, nIm_res, 3)))

                if training_step % log_every_n_steps == 0:
                    # print("Iteration {}, batch: {} generator loss: {:.3f} ({:.3f}), discriminator loss: {:.3f}".
                    #       format(num_itr, batch, gen_loss.numpy(), gen_loss_adv.numpy(), dis_loss.numpy()))

                    print("Iteration {}, batch: {} Total generator loss: {:.3f} (appearance : {:.3f} - normal : {:.3f} - env: {:.3f}) ".
                          format(num_itr, batch,
                                 log_losses_ratio * gen_loss.numpy(), log_losses_ratio * loss_app.numpy(),
                                 log_losses_ratio * loss_norm.numpy(), log_losses_ratio * loss_env.numpy()))
                    if (perform_testing):
                        print ("Testing loss : {:.3f} ".format(testing_loss))
                if training_step % write_every_n_steps == 0:
                    save_tensor_cv2(appearances_preds[0], result_training_folder_name + "Appearance.png")
                    save_tensor_cv2(envMap_preds[0], result_training_folder_name + "EnvMap.hdr", 1)
                    save_tensor_cv2(normals_preds[0], result_training_folder_name + "Normal.png")
                    save_tensor_cv2(tf.pow(albedos_preds[0],invGamma), result_training_folder_name + "Albedo.png")

        num_itr = num_itr + 1


    root.save(checkpoint_write_prefix)
    return

if __name__ == "__main__":
    main()
