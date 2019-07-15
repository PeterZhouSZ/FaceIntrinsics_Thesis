from __future__ import absolute_import, division, print_function

from generator_V1 import FirstGenerator
from autoencoder_convolutional_V2 import SecondUVAutoencoder
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imageio
import envMapOrientation
import normalMapProcessing
import cv2
from load_rgb_cv import load_rgb, load_rgba,convert_rgb_to_cv2
from data_loader import load_envMaps

predict_sphere_envMap = (True,False)  # first argument : predict - second : sphere
high_res_mode = True
albedo_boosting_factor = 1. / 1. #0.65
envDir = "EnvMaps/"
envName = "village"  # field field2 field3 field4 field5 park Uffizi village harbour_bright village2
if (predict_sphere_envMap[1]):
    envFile = envDir + envName + "_probe.hdr"
else:
    envFile = envDir + envName + ".hdr"

input_data_folder = "/Users/benjaminbarral/Documents/Academic/UCL/Research Project/3DMMExpe/TexturesChicagoCorrected/"#_Appearance_UV.png"
output_data_folder = "Synthesized_Data/TobiasMail/"

code = "CFD-AF-207-023-N"
code_write = ""
albedo_file = input_data_folder + code + "_Appearance_UV.png"
normal_file = input_data_folder + code + "_Normal_UV.png"
albedoMap = load_rgba(albedo_file)  # "Data02_isomaps/AppearanceMap_test.png")
normalMap = load_rgb(normal_file)  # "Data02_isomaps/NormalMap_test.png")

data_input_folder_name = "Data_to_synthesize"

input_shape = (512, 512, 3)
num_replicate = 1

envMap = load_rgb(envFile, -1)
print("Before resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap), np.mean(envMap)))

(mEnv, nEnv, dEnv) = envMap.shape
print((mEnv, nEnv))
plt.imshow(envMap)
plt.show()
albedoMap = cv2.resize(albedoMap, (input_shape[0], input_shape[1]))
normalMap = cv2.resize(normalMap, (input_shape[0], input_shape[1]))
envMap = cv2.resize(envMap, (64, 32), interpolation=cv2.INTER_LINEAR)

print("After resize : max: {:.3f}, mean: ({:.3f})".format(np.amax(envMap), np.mean(envMap)))

(mIm, nIm, dIm) = albedoMap.shape
(mEnv, nEnv, dEnv) = envMap.shape
input_shape = albedoMap.shape
d = 3

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
albedoTensor = tf.constant(albedoMap[:, :, :3], dtype=tf.float32)
normalTensor = tf.constant(normalMap[:, :, :3], dtype=tf.float32)
envMapTensor = tf.constant(envMap, dtype=tf.float32)
albedoTensor = tf.scalar_mul(1. / normalizingValue, albedoTensor[:, :, :3])
normalTensor = tf.scalar_mul(1. / normalizingValue, normalTensor[:, :, :3])

# Boost albedo
albedo_boosted_Tensor = tf.scalar_mul(tf.constant(albedo_boosting_factor),tf.pow(albedoTensor,gamma))


plt.imshow(envMapTensor)
plt.show()
plt.imshow(albedoTensor)
plt.show()
plt.imshow(albedo_boosted_Tensor)
plt.show()
# plt.imshow(normalTensor)
# plt.show()


if (predict_sphere_envMap[1]):
    envNormalization = tf.constant(math.pi * mEnv * nEnv / 4.)
else:
    envNormalization = tf.constant((float)(mEnv * nEnv))

## Calculate envMap orientations
envVectors = envMapOrientation.envMapOrientation(mEnv, nEnv, predict_sphere_envMap[1])
envOrientationTensor = tf.constant(envVectors, dtype=tf.float32)
envOrientationTensor = tf.reshape(envOrientationTensor, [3, mEnv * nEnv])
envMapTensor = tf.reshape(envMapTensor, [mEnv * nEnv, 3])

autoencoder_model = FirstGenerator(input_shape, envOrientationTensor,
                                   envMapTensor, envNormalization, predict_sphere_envMap, high_res_mode)

normalTensor = tf.reshape(normalTensor, [1, mIm, nIm, d])
albedoTensor = tf.reshape(albedoTensor, [1, mIm, nIm, d])

# resTensor = autoencoder_model.render(normalTensor,albedoTensor)
resTensor = autoencoder_model.render_with_predicted_envMap(normalTensor, albedoTensor,
                                                           tf.reshape(envMapTensor,
                                                                      [num_replicate, mEnv * nEnv, 3]))

resTensorGamma = tf.pow(resTensor, invGamma)

res_save = 255. * np.array(resTensorGamma[0])
res_save = convert_rgb_to_cv2(res_save)
res_save_temp = np.zeros([res_save.shape[0], res_save.shape[1], 4])
res_save_temp[:, :, :3] = res_save[:, :, :]
res_save_temp[:, :, 3] = albedoMap[:, :, 3]
res_save = res_save_temp.astype(int)

plt.imshow(resTensor[0])
plt.show()
plt.imshow(resTensorGamma[0])
plt.show()
#cv2.imwrite(output_data_folder+ code_write + "_Appearance_UV.png", res_save)

resTensorBoosted = autoencoder_model.render_with_predicted_envMap(normalTensor, albedo_boosted_Tensor,
                                                           tf.reshape(envMapTensor,
                                                                      [num_replicate, mEnv * nEnv, 3]))

resTensorBoostedGamma = tf.pow(resTensorBoosted, invGamma)

res_save = 255. * np.array(resTensorBoostedGamma[0])
res_save = convert_rgb_to_cv2(res_save)
res_save_temp = np.zeros([res_save.shape[0], res_save.shape[1], 4])
res_save_temp[:, :, :3] = res_save[:, :, :]
res_save_temp[:, :, 3] = albedoMap[:, :, 3]
res_save = res_save_temp.astype(int)

plt.imshow(resTensorBoosted[0])
plt.show()
plt.imshow(resTensorBoostedGamma[0])
plt.show()
cv2.imwrite(output_data_folder+ code_write + "_Appearance_UV.png", res_save)


