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

sphereProbeMode = True
def main():
    plt.close("all")
    tf.enable_eager_execution()

    envDir = "EnvMaps/"
    envName =envDir +  "uffizi"
    if (sphereProbeMode):
        envFile = envName +"_probe.hdr"
    else :
        envFile = envName + ".hdr"
    albedoMap = imageio.imread("venv/Albedo.png") #"Data02_isomaps/AppearanceMap_test.png")
    normalMap = imageio.imread("venv/Normal.png") #"Data02_isomaps/NormalMap_test.png")
    envMap = imageio.imread(envFile, format='HDR-FI')

    #print(np.amax(envMap))

    resize_rate = 0.05
    (mIm, nIm, dIm) = albedoMap.shape
    (mEnv, nEnv, dEnv) = envMap.shape
    input_shape = albedoMap.shape
    d = 3

    normalProcessed = normalMapProcessing.processNormalMap(normalMap, mIm, nIm)
    normalizingValue = tf.constant(255.)
    albedoTensor = tf.constant(albedoMap[:,:,:3], dtype=tf.float32)
    normalTensor = tf.constant(normalProcessed[:,:,:3], dtype=tf.float32)
    envMapTensor = tf.constant(envMap)
    albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])

    mEnv = int(resize_rate * mEnv)
    nEnv = int(resize_rate * nEnv)
    envMapTensor = tf.image.resize(envMapTensor, tf.constant([mEnv, nEnv]))
    envMapTensorBis = tf.reshape(envMapTensor, [1, mEnv * nEnv, 3])
    plt.imshow(envMapTensor)
    plt.show()
    envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])
    print((mEnv, nEnv))

    if (sphereProbeMode):
        envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
    else :
        envNormalization = tf.constant((float)(mEnv * nEnv))

    ## Calculate envMap orientations
    envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, sphereProbeMode)
    envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
    envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])

    autoencoder_model = FirstUVAutoencoder(input_shape, envOrientationTensor, envMapTensor, envNormalization, (False,False),False)

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