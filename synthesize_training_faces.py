from __future__ import absolute_import, division, print_function

from generator_V1 import FirstGenerator
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import math
import envMapOrientation
import normalMapProcessing
import cv2
from load_rgb_cv import load_rgb, load_rgba,convert_rgb_to_cv2
import shutil

predict_sphere_envMap = (True,False)  # first argument : predict - second : sphere
high_res_mode = True
albedo_mode = 1
albedo_boosting_factor = 1. / 0.65
gamma = tf.constant(2.2)
invGamma = tf.constant(1. / 2.2)
normalizingValue = tf.constant(255.)

envmap_Size = (32, 64)
face_size = (256,256)

(mEnv,nEnv) = envmap_Size
(mIm,nIm) = face_size
input_shape = (mIm,nIm,3)

if (predict_sphere_envMap[1]):
    envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
else:
    envNormalization = tf.constant((float)(mEnv * nEnv))

## Calculate envMap orientations
envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, predict_sphere_envMap[1])
envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])

data_folder = "Synthesized_Data/"
input_data_folder = data_folder + "Input/"
output_data_folder = data_folder + "Output/"
envDir = "EnvMaps/"
envFiles = ["field", "field2", "field3", "field4", "field5", "park", "Uffizi", "village", "harbour_bright", "village2"]
input_codes = ["F6", "F7"]

def render_face(envName, input_code, output_code, log_and_show, write_results, albedo_mode):
    if (predict_sphere_envMap[1]):
        envFile = envDir + envName + "_probe.hdr"
    else:
        envFile = envDir + envName + ".hdr"

    albedo_file = input_data_folder + input_code + "_Appearance_UV.png"
    normal_file = input_data_folder + input_code + "_Normal_UV.png"
    # LOAD
    albedoMap = load_rgba(albedo_file)  # "Data02_isomaps/AppearanceMap_test.png")
    normalMap = load_rgb(normal_file)  # "Data02_isomaps/NormalMap_test.png")
    envMap = load_rgb(envFile, -1)
    if (write_results):
        shutil.copy(envFile, output_data_folder + output_code + "_Illum.hdr")

    (mEnv_input, nEnv_input, dEnv) = envMap.shape

    if (log_and_show):
        print("Before resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap), np.mean(envMap)))
        print("Input envmap size : ({:},{:})".format(mEnv_input, nEnv_input))
        plt.imshow(envMap)
        plt.show()

    #RESIZE
    albedoMap = cv2.resize(albedoMap, (nIm, mIm))
    normalMap = cv2.resize(normalMap, (nIm, mIm))
    envMap = cv2.resize(envMap, (nEnv,mEnv) , interpolation=cv2.INTER_LINEAR)

    if (log_and_show):
        print("After resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap), np.mean(envMap)))

    d = 3

    albedoTensor = tf.constant(albedoMap[:, :, :3], dtype=tf.float32)
    normalTensor = tf.constant(normalMap[:, :, :3], dtype=tf.float32)
    envMapTensor = tf.constant(envMap, dtype=tf.float32)
    albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])
    normalTensor = tf.scalar_mul(1. / normalizingValue, normalTensor[:, :, :3])



    # Boost albedo
    albedo_boosted_Tensor = tf.scalar_mul(tf.constant(albedo_boosting_factor), tf.pow(albedoTensor, gamma))

    if (log_and_show):
        plt.imshow(envMapTensor)
        plt.show()
        plt.imshow(albedoTensor)
        plt.show()
        plt.imshow(tf.pow(albedo_boosted_Tensor, invGamma))
        plt.show()
        # plt.imshow(normalTensor)
        # plt.show()

    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])
    normalTensor = tf.reshape(normalTensor, [1, mIm, nIm, d])
    albedoTensor = tf.reshape(albedoTensor, [1, mIm, nIm, d])


    autoencoder_model = FirstGenerator(input_shape, envOrientationTensor,
                                       envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)


    # resTensor = autoencoder_model.render(normalTensor,albedoTensor)
    if albedo_mode==0:
        albedo_render = albedoTensor
        albedo_save = albedoMap
    else:
        albedo_render = albedo_boosted_Tensor
        albedo_save = 255. * np.array(tf.pow(albedo_boosted_Tensor, invGamma))

    resTensor = autoencoder_model.render_with_predicted_envMap(normalTensor, albedo_render,
                                                               tf.reshape(envMapTensor,
                                                                          [1, mEnv * nEnv, 3]))

    resTensorGamma = tf.pow(resTensor, invGamma)
    res_save = 255. * np.array(resTensorGamma[0])
    res_save = convert_rgb_to_cv2(res_save)
    res_save_temp = np.zeros([res_save.shape[0], res_save.shape[1], 4])
    res_save_temp[:, :, :3] = res_save[:, :, :]
    res_save_temp[:, :, 3] = albedoMap[:, :, 3]
    res_save = res_save_temp.astype(int)

    albedo_save = convert_rgb_to_cv2(albedo_save)
    albedo_save = albedo_save.astype(int)

    if (log_and_show):
        # plt.imshow(resTensor[0])
        # plt.show()
        plt.imshow(resTensorGamma[0])
        plt.show()

    if (write_results):
        cv2.imwrite(output_data_folder+ output_code + "_Appearance_UV.png", res_save)
        cv2.imwrite(output_data_folder+ output_code + "_Albedo_UV.png", albedo_save)
        cv2.imwrite(output_data_folder+ output_code + "_Normal_UV.png", normalMap)

    return

for input_code in input_codes:
    illum_code = 1
    for envFile in envFiles:
        output_code = input_code + "_I" + str(illum_code)
        envName = envFile
        render_face(envName, input_code, output_code, True, True, 1)
        illum_code +=1





