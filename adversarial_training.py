from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from load_rgb_cv import load_rgb,convert_rgb_to_cv2

import numpy as np
from data_loader import load_input_data_with_normals_and_replicate, load_ground_truth_data, plot_data_bis
from generator_V1 import FirstGenerator
from discriminator_V1 import FirstDiscriminator
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

show_every_n_steps = 1 # tensorboard
log_every_n_steps = 1

lambda_adv = .01

learning_rate_disc = 1e-4

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
    data_input_folder_name = "Data04_isomaps_normals/"
    data_ground_truth_folder_name = "TexturesChicago/"
    ratio_train = 0.75
    max_files = 5000
    (dataset_adv_real, num_adv_real) = load_ground_truth_data(data_ground_truth_folder_name,
                                                              resize_images= True, image_shape= shape_gt_adv)
    (dataset_train_input, num_train_input) = load_input_data_with_normals_and_replicate(data_input_folder_name, input_shape,True, num_adv_real)

    plot_data_bis(dataset_train_input, 0)
    plot_data_bis(dataset_train_input, 3)
    plot_data_bis(dataset_train_input, 2)



    epochs = 50
    batch_size = 5
    num_batches = int(num_train_input / batch_size)

    envDir = "EnvMaps/"
    envName = envDir + "village"
    envFile = envName + "_probe.hdr"
    envMap = load_rgb(envName,-1)
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

    checkpoint_write_dir = "model_saved"#_10_samples_withEnvMap_OkResults/"
    checkpoint_load_dir = "model_10_samples_withEnvMap_OkResults/"
    checkpoint_write_prefix = checkpoint_write_dir + "adversarial"

    #autoencoder_model = SecondUVAutoencoder(input_shape, envOrientationTensor,envMapTensor, envNormalization, predict_envMap)
    generator_model = FirstGenerator(input_shape, envOrientationTensor,
                                           envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)
    discriminator_model = FirstDiscriminator()

    generator_optimizer = tf.train.AdamOptimizer()
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate_disc)

    logdir = "./tb/"
    global_step = tf.train.get_or_create_global_step()
    summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10000)

    root = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer = discriminator_optimizer,
                               generator_model = generator_model,
                               discriminator_model = discriminator_model,
                               optimizer_step=tf.train.get_or_create_global_step())

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
    while (num_itr < epochs):
        dataset_train_input = dataset_train_input.shuffle(num_train_input)
        x_train_input = dataset_train_input.batch(batch_size)
        dataset_adv_real = dataset_adv_real.shuffle(num_adv_real)
        x_train_adv_real = dataset_adv_real.batch(batch_size)
        for (batch, ((inputs, labels_appearance, labels_normals, masks), ground_truth_images)) \
                in enumerate(zip(x_train_input,x_train_adv_real)):
            #plt.imshow(ground_truth_images[0])
            #plt.show()
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    (albedos_preds,normals_preds,appearances_preds, envMap_preds) = generator_model(inputs)

                    fake_adv_output = discriminator_model(appearances_preds)
                    real_adv_output = discriminator_model(ground_truth_images)

                    gen_loss_l2 = generator_model.loss(appearances_preds, normals_preds,
                                                  albedos_preds,labels_appearance, labels_normals,masks)
                    gen_loss_adv = discriminator_model.generator_loss(fake_adv_output)

                    gen_loss = gen_loss_l2 + lambda_adv * gen_loss_adv
                    dis_loss = discriminator_model.discriminator_loss(real_adv_output, fake_adv_output)


                gradients_of_generator = gen_tape.gradient(gen_loss, generator_model.variables)
                gradients_of_discriminator = disc_tape.gradient(dis_loss, discriminator_model.variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                            discriminator_model.variables),
                                                        global_step=tf.train.get_or_create_global_step())



                if (batch % show_every_n_steps == 0):
                    tf.contrib.summary.scalar("generator total loss", gen_loss)
                    tf.contrib.summary.scalar("adversarial loss (generator)", gen_loss_adv)
                    tf.contrib.summary.scalar("adversarial loss (discriminator)", dis_loss)
                    tf.contrib.summary.image("appearance input", tf.reshape(inputs[0], (1, mIm, nIm, 3)))
                    tf.contrib.summary.image("normal map (from 3DMM) ", tf.reshape(labels_normals[0], (1, mIm, nIm, 3)))
                    tf.contrib.summary.image("albedo prediction", tf.reshape(albedos_preds[0],
                                                                             (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("normal map prediction",
                                             tf.reshape(normals_preds[0],
                                                        (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("appearance result",
                                             tf.reshape(appearances_preds[0],
                                                        (1, mIm_res, nIm_res, 3)))
                    tf.contrib.summary.image("ground truth high res texture",
                                             tf.reshape(ground_truth_images[0],
                                                        (1, mIm_res, nIm_res, 3)))
                    if (predict_sphere_envMap[0]):
                        if (predict_sphere_envMap[1]):
                            envMap_show = tf.multiply(envMapMask, envMap_preds[0])
                        else:
                            envMap_show = envMap_preds[0]
                        tf.contrib.summary.image("envMap result", tf.reshape(envMap_show, (1, mEnv, nEnv, 3)))
                if batch % log_every_n_steps == 0:
                    print("Iteration {}, batch: {} generator loss: {:.3f} ({:.3f}), discriminator loss: {:.3f}".
                          format(num_itr, batch, gen_loss.numpy(), gen_loss_adv.numpy(), dis_loss.numpy()))
        num_itr = num_itr + 1


    root.save(checkpoint_write_prefix)
    return

if __name__ == "__main__":
    main()
