import cv2
import tensorflow as tf
import numpy as np

def load_rgba(path):
    temp = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    b, g, r, a = cv2.split(temp)  # get b,g,r
    res = cv2.merge([r, g, b, a])
    return res

def load_rgb(path, code= 1):
    if (code==-1):
        temp = cv2.imread(path,-1)
    else:
        temp = cv2.imread(path)
    b, g, r = cv2.split(temp)  # get b,g,r
    res = cv2.merge([r, g, b])
    return res

def convert_rgb_to_cv2(im):
    r, g, b = cv2.split(im)
    res = cv2.merge([b, g, r])
    return res

def tonemap(imTensor, gamma):
    temp = tf.clip_by_value(imTensor, 0, 1)
    #temp = tf.multiply(1./tf.reduce_max(imTensor),imTensor)
    #temp = tf.pow(temp,gamma)
    return temp

def save_tensor_cv2(tensor, path, envMap = -1):
    if (envMap == -1):
        res_save = 255. * np.array(tensor)
        res_save = convert_rgb_to_cv2(res_save)
        res_save = res_save.astype(int)
    else: # write HDR
        res_save = np.array(tensor)
        res_save = convert_rgb_to_cv2(res_save)
    cv2.imwrite(path, res_save)
    return
