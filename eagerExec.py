from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import math
import envMapOrientation
import normalMapProcessing

def main():
    plt.close("all")
    tf.enable_eager_execution()

    envName = "kitchen"
    envFile = envName + "_probe.hdr"
    albedoMap = imageio.imread("../Data02_isomaps/AppearanceMap_test.png") #("Albedo.png")
    normalMap = imageio.imread("../Data02_isomaps/NormalMap_test.png") #("Normal.png")
    envMap = imageio.imread(envName + '_probe.hdr', format='HDR-FI')

    resize_rate = 0.05
    (mIm, nIm, dIm) = albedoMap.shape
    (mEnv, nEnv, dEnv) = envMap.shape
    d = 3

    normalProcessed = normalMapProcessing.processNormalMap(normalMap, mIm, nIm)
    normalizingValue = tf.constant(255.)
    albedoTensor = tf.constant(albedoMap, dtype=tf.float32)
    normalTensor = tf.constant(normalProcessed, dtype=tf.float32)
    envMapTensor = tf.constant(envMap)
    albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])

    mEnv = int(resize_rate * mEnv)
    nEnv = int(resize_rate * nEnv)
    envMapTensor = tf.image.resize(envMapTensor, tf.constant([mEnv, nEnv]))
    print((mEnv, nEnv))

    plt.imshow(envMapTensor)
    plt.show()

    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv)
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)

    # normalTensor = tf.scalar_mul(1./normalizingValue, normalTensor)
    # normalTensor = (normalTensor - 0.5 ) * 2.

    normalTensor = tf.reshape(normalTensor, [mIm * nIm, 3])
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])

    cosineTensor = tf.matmul(normalTensor, envOrientationTensor)
    cosineTensor = tf.clip_by_value(cosineTensor, 0, 1)
    # cosineTensor = tf.reshape(cosineTensor,[mIm,nIm,mEnv * nEnv])
    # resTensor = tf.get_variable("resTensor", (mIm,nIm,d),dtype = tf.float32,initializer = tf.zeros_initializer())

    envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)

    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])
    shadedBrightness = tf.matmul(cosineTensor, envMapTensor)
    shadedBrightness = tf.scalar_mul(1. / envNormalization, tf.reshape(shadedBrightness, [mIm, nIm, 3]))

    # unshadedBrightness = tf.scalar_mul(1./envNormalization,tf.reduce_sum(envMapTensor,[0,1]))

    gamma = 2.2
    resTensor = tf.multiply(albedoTensor, shadedBrightness)
    resTensorTM = tf.pow(resTensor, gamma)
    resTensor = tf.scalar_mul(normalizingValue, resTensor)
    resTensorTM = tf.scalar_mul(normalizingValue, resTensorTM)

    resIm = np.array(resTensor)
    resImTM = np.array(resTensorTM)
    plt.imshow(resIm.astype(int))
    plt.show()
    plt.imshow(resImTM.astype(int))
    plt.show()

    imageio.imsave("result_" + envName + ".jpg", resIm)



if __name__ == "__main__":
    main()