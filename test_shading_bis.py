from __future__ import absolute_import, division, print_function
from autoencoder_convolutional_V1 import FirstUVAutoencoder
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import math
import envMapOrientation
import normalMapProcessing

sphereProbeMode = False
def main():
    plt.close("all")
    tf.enable_eager_execution()

    envDir = "EnvMaps/"
    envName =envDir +  "uffizi"
    envFile1 = envName +"_probe.hdr"
    envFile2 = envName + ".hdr"
    albedoMap = imageio.imread("venv/Albedo.png") #"Data02_isomaps/AppearanceMap_test.png")
    normalMap = imageio.imread("venv/Normal.png") #"Data02_isomaps/NormalMap_test.png")
    envMap1 = imageio.imread(envFile1, format='HDR-FI')
    envMap2 = imageio.imread(envFile2, format='HDR-FI')

    #print(np.amax(envMap))

    resize_rate_1 = 0.05
    resize_rate_2 = 0.075
    (mIm, nIm, dIm) = albedoMap.shape
    (mEnv1, nEnv1, dEnv1) = envMap1.shape
    (mEnv2, nEnv2, dEnv2) = envMap1.shape
    input_shape = albedoMap.shape
    d = 3

    normalProcessed = normalMapProcessing.processNormalMap(normalMap, mIm, nIm)
    normalizingValue = tf.constant(255.)
    albedoTensor = tf.constant(albedoMap[:,:,:3], dtype=tf.float32)
    normalTensor = tf.constant(normalProcessed[:,:,:3], dtype=tf.float32)
    envMapTensor1 = tf.constant(envMap1)
    envMapTensor2 = tf.constant(envMap2)
    albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])

    mEnv1 = int(resize_rate_1 * mEnv1)
    nEnv1 = int(resize_rate_1 * nEnv1)
    mEnv2 = int(resize_rate_2 * mEnv2)
    nEnv2 = int(resize_rate_2 * nEnv2)
    envMapTensor1 = tf.image.resize(envMapTensor1, tf.constant([mEnv1, nEnv1]))
    envMapTensor2 = tf.image.resize(envMapTensor2, tf.constant([mEnv2, nEnv2]))


    ## Calculate envMap orientations
    envVectors1 = envMapOrientation.envMapOrientationDebug(mEnv1, nEnv1, True)
    envVectors2 = envMapOrientation.envMapOrientationDebug(mEnv2, nEnv2, False)
    envOrientationTensor1 = tf.constant(envVectors1, dtype=tf.float32)
    envOrientationTensor2 = tf.constant(envVectors2, dtype=tf.float32)

    debug_1 = tf.multiply(envOrientationTensor1, envMapTensor1)
    debug_2 = tf.multiply(envOrientationTensor2, envMapTensor2)

    plt.imshow(envOrientationTensor1)
    plt.show()
    plt.imshow(debug_1)
    plt.show()
    plt.imshow(envOrientationTensor2)
    plt.show()
    plt.imshow(debug_2)
    plt.show()

    return;
    autoencoder_model = FirstUVAutoencoder(input_shape, envOrientationTensor, envMapTensor, envNormalization, False)

    normalTensor = tf.reshape(normalTensor, [1,mIm, nIm, d])
    albedoTensor = tf.reshape(albedoTensor, [1, mIm, nIm, d])

    resTensor = autoencoder_model.render(normalTensor,albedoTensor)
    #resTensor = autoencoder_model.render_with_predicted_envMap(normalTensor, albedoTensor, envMapTensorBis)

    plt.imshow(resTensor[0])
    plt.show()
    return

    # normalTensor = tf.scalar_mul(1./normalizingValue, normalTensor)
    # normalTensor = (normalTensor - 0.5 ) * 2.

    normalTensor = tf.reshape(normalTensor, [mIm * nIm, 3])
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])

    cosineTensor = tf.matmul(normalTensor, envOrientationTensor)
    cosineTensor = tf.clip_by_value(cosineTensor, 0, 1)
    # cosineTensor = tf.reshape(cosineTensor,[mIm,nIm,mEnv * nEnv])
    # resTensor = tf.get_variable("resTensor", (mIm,nIm,d),dtype = tf.float32,initializer = tf.zeros_initializer())

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