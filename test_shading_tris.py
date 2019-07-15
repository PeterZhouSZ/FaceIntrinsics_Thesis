from __future__ import absolute_import, division, print_function
from generator_V1 import FirstGenerator
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import numpy as np
import tensorflow as tf
#import scipy.misc as spm
import matplotlib.pyplot as plt
import imageio
import math
import envMapOrientation
import normalMapProcessing
import cv2
from load_rgb_cv import load_rgb,convert_rgb_to_cv2
import shutil

predict_sphere_envMap = (True,False)  # first argument : predict - second : sphere
high_res_mode = True
def main():
    plt.close("all")
    tf.enable_eager_execution()

    folder_write_name = "Data_synthesized/"
    extension_nor_app = "AppearanceNormal/"
    extension_env = "EnvMap/"
    code_sample = "S4_I3_"


    envDir = "EnvMaps/"
    envName =envDir +  "STUDIOATM_13SN"
    if (predict_sphere_envMap[1]):
        envFile = envName +"_probe.hdr"
    else :
        #envFile = folder_write_name + extension_env + code_sample + "Illum.hdr"
        envFile = envName + ".hdr"
    albedoMap = load_rgb("Chicago_albedo.png") #"Data02_isomaps/AppearanceMap_test.png")
    normalMap = load_rgb("Chicago_normal.png") #"Data02_isomaps/NormalMap_test.png")
    envMap = load_rgb(envFile, -1)
    print("Before resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap),np.mean(envMap)))

    shutil.copy("Chicago_normal.png",folder_write_name + extension_nor_app +
                 code_sample + "Normal_UV.png")
    shutil.copy(envFile,folder_write_name + extension_env +
                 code_sample + "Illum.hdr")


    (mEnv,nEnv,dEnv) = envMap.shape
    print((mEnv,nEnv))
    plt.imshow(envMap)
    plt.show()
    albedoMap = cv2.resize(albedoMap, (256,256))
    normalMap = cv2.resize(normalMap, (256,256))
    envMap = cv2.resize(envMap,(3*32,3*16), interpolation = cv2.INTER_LINEAR)

    print("After resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap),np.mean(envMap)))

    (mIm, nIm, dIm) = albedoMap.shape
    (mEnv, nEnv, dEnv) = envMap.shape
    input_shape = albedoMap.shape
    d = 3

    plt.imshow(envMap)
    plt.show()
    plt.imshow(albedoMap)
    plt.show()
    plt.imshow(normalMap)
    plt.show()

    # cv2.namedWindow('albedo', cv2.WINDOW_NORMAL)
    # cv2.imshow('albedo', albedoMap)
    # cv2.namedWindow('normal', cv2.WINDOW_NORMAL)
    # cv2.imshow('normal', normalMap)
    # cv2.namedWindow('envMap', cv2.WINDOW_NORMAL)
    # cv2.imshow('envMap', envMap)
    # cv2.waitKey(0)


    gamma = tf.constant(2.2)
    invGamma = tf.constant(1. / 2.2)
    normalizingValue = tf.constant(255.)
    albedoTensor = tf.constant(albedoMap[:,:,:3], dtype=tf.float32)
    normalTensor = tf.constant(normalMap[:,:,:3], dtype=tf.float32)
    envMapTensor = tf.constant(envMap, dtype=tf.float32)
    albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])
    normalTensor = tf.scalar_mul(1. / normalizingValue, normalTensor[:, :, :3])
    #albedoTensor = tf.pow(albedoTensor,gamma)

    if (predict_sphere_envMap[1]):
        envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
    else :
        envNormalization = tf.constant((float)(mEnv * nEnv))

    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, predict_sphere_envMap[1])
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])
    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])

    autoencoder_model = FirstGenerator(input_shape, envOrientationTensor,
                                           envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)

    normalTensor = tf.reshape(normalTensor, [1,mIm, nIm, d])
    albedoTensor = tf.reshape(albedoTensor, [1, mIm, nIm, d])

    resTensor = autoencoder_model.render(normalTensor,albedoTensor)
    resTensorGamma = tf.pow(resTensor,invGamma)

    res_save = 255. * np.array(resTensorGamma[0])
    res_save = convert_rgb_to_cv2(res_save)
    res_save = res_save.astype(int)

    plt.imshow(resTensor[0])
    plt.show()
    plt.imshow(resTensorGamma[0])
    plt.show()

    #cv2.imwrite(folder_name+ "envMap_resized.hdr" , envMap)
    cv2.imwrite(folder_write_name+extension_nor_app+code_sample+ "Appearance_UV.png" , res_save)
    #cv2.imwrite(folder_name+ "normalMap_resized.png" , normalMap)

    return



if __name__ == "__main__":
    main()