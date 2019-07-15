from __future__ import absolute_import, division, print_function
import pathlib
import random
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
tf.enable_eager_execution()
import cv2

from load_rgb_cv import load_rgb, load_rgba,convert_rgb_to_cv2

def load_data(data_folder_name, image_shape, ratio_train, max_files, alpha_channel):
    data_root = pathlib.Path(data_folder_name)
    all_image_paths = list(data_root.glob('*.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    random.shuffle(all_image_paths)

    image_count = len(all_image_paths)

    if (image_count > max_files):
        train_count = int (ratio_train * max_files)
        train_paths = all_image_paths[:train_count]
        test_paths = all_image_paths[train_count:max_files]
        num_train = train_count
        num_test = max_files - train_count

    else :
        train_count = int(ratio_train * image_count)
        train_paths = all_image_paths[:train_count]
        test_paths = all_image_paths[train_count:]
        num_train = train_count
        num_test = image_count - train_count

    train_paths_ds = tf.data.Dataset.from_tensor_slices(train_paths)
    test_paths_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0  # normalize to [0,1] range
        return image

    def create_mask(path):
        image = tf.read_file(path)
        if alpha_channel:
            image = tf.image.decode_jpeg(image, channels=4)
        else:
            image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0
        if alpha_channel:
            alpha = image[:, :, 3]
            near_zero_tensor = tf.constant(0.01, dtype=tf.float32, shape=(image_shape[0], image_shape[1]))
            masked_a = tf.where(tf.greater(alpha, near_zero_tensor), tf.ones_like(alpha), tf.zeros_like(alpha))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a
        else:
            grey = tf.image.rgb_to_grayscale(image)
            near_zero_tensor = tf.constant(0.05, dtype=tf.float32, shape=(image_shape[0], image_shape[1],1))
            masked_a = tf.where(tf.greater(grey, near_zero_tensor), tf.ones_like(grey), tf.zeros_like(grey))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a


    train_image_ds = train_paths_ds.map(load_and_preprocess_image)
    test_image_ds = test_paths_ds.map(load_and_preprocess_image)
    mask_train_ds = train_paths_ds.map(create_mask)
    mask_test_ds = test_paths_ds.map(create_mask)

    train_dataset = tf.data.Dataset.zip((train_image_ds,train_image_ds, mask_train_ds))
    test_dataset = tf.data.Dataset.zip((test_image_ds,test_image_ds, mask_test_ds))

    return (train_dataset, num_train, test_dataset, num_test)

def load_input_data_with_normals(data_folder_name, image_shape, alpha_channel):
    data_root = pathlib.Path(data_folder_name)
    all_image_paths = list(data_root.glob('*Appearance_UV.png'))
    all_normal_paths = list(data_root.glob('*Normal_UV.png'))
    all_image_paths = [str(path) for path in all_image_paths]
    all_normal_paths = [str(path) for path in all_normal_paths]

    all_image_paths.sort()
    all_normal_paths.sort()

    image_count = len(all_image_paths)
    num_train = image_count

    image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    normal_paths_ds = tf.data.Dataset.from_tensor_slices(all_normal_paths)

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0  # normalize to [0,1] range
        return image

    def create_mask(path):
        image = tf.read_file(path)
        if alpha_channel:
            image = tf.image.decode_jpeg(image, channels=4)
        else:
            image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0
        if alpha_channel:
            alpha = image[:, :, 3]
            near_zero_tensor = tf.constant(0.01, dtype=tf.float32, shape=(image_shape[0], image_shape[1]))
            masked_a = tf.where(tf.greater(alpha, near_zero_tensor), tf.ones_like(alpha), tf.zeros_like(alpha))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a
        else:
            grey = tf.image.rgb_to_grayscale(image)
            near_zero_tensor = tf.constant(0.05, dtype=tf.float32, shape=(image_shape[0], image_shape[1],1))
            masked_a = tf.where(tf.greater(grey, near_zero_tensor), tf.ones_like(grey), tf.zeros_like(grey))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a


    train_image_ds = image_paths_ds.map(load_and_preprocess_image)
    train_normal_ds = normal_paths_ds.map(load_and_preprocess_image)
    mask_train_ds = image_paths_ds.map(create_mask)

    train_dataset = tf.data.Dataset.zip((train_image_ds,train_image_ds,train_normal_ds, mask_train_ds))

    return (train_dataset, num_train)

def load_input_data_with_normals_and_replicate(data_folder_name, image_shape, alpha_channel,num_total):
    data_root = pathlib.Path(data_folder_name)
    all_image_paths = list(data_root.glob('*Appearance_UV.png'))
    all_normal_paths = list(data_root.glob('*Normal_UV.png'))
    path_im = all_image_paths[0]
    path_nor = all_normal_paths[0]
    all_image_paths = [str(path_im) for i in range(num_total)]
    all_normal_paths = [str(path_nor) for i in range(num_total)]

    all_image_paths.sort()
    all_normal_paths.sort()

    image_count = len(all_image_paths)
    num_train = image_count

    image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    normal_paths_ds = tf.data.Dataset.from_tensor_slices(all_normal_paths)

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0  # normalize to [0,1] range
        return image

    def create_mask(path):
        image = tf.read_file(path)
        if alpha_channel:
            image = tf.image.decode_jpeg(image, channels=4)
        else:
            image = tf.image.decode_jpeg(image, channels=3)

        image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0
        if alpha_channel:
            alpha = image[:, :, 3]
            near_zero_tensor = tf.constant(0.01, dtype=tf.float32, shape=(image_shape[0], image_shape[1]))
            masked_a = tf.where(tf.greater(alpha, near_zero_tensor), tf.ones_like(alpha), tf.zeros_like(alpha))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a
        else:
            grey = tf.image.rgb_to_grayscale(image)
            near_zero_tensor = tf.constant(0.05, dtype=tf.float32, shape=(image_shape[0], image_shape[1],1))
            masked_a = tf.where(tf.greater(grey, near_zero_tensor), tf.ones_like(grey), tf.zeros_like(grey))
            masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
            masked_a = tf.tile(masked_a, [1, 1, 3])
            return masked_a


    train_image_ds = image_paths_ds.map(load_and_preprocess_image)
    train_normal_ds = normal_paths_ds.map(load_and_preprocess_image)
    mask_train_ds = image_paths_ds.map(create_mask)

    train_dataset = tf.data.Dataset.zip((train_image_ds,train_image_ds,train_normal_ds, mask_train_ds))

    return (train_dataset, num_train)

def load_input_data_with_albedo_and_envmaps(input_folder_name,
                                            envMap_shape,
                                            image_shape, alpha_channel):
    input_data_root = pathlib.Path(input_folder_name)
    all_image_paths = list(input_data_root.glob('*Appearance_UV.png'))
    all_normal_paths = list(input_data_root.glob('*Normal_UV.png'))
    all_albedo_paths = list(input_data_root.glob('*Albedo_UV.png'))
    all_env_paths = list(input_data_root.glob('*Illum.hdr'))

    all_image_paths = [str(path) for path in all_image_paths]
    all_normal_paths = [str(path) for path in all_normal_paths]
    all_albedo_paths = [str(path) for path in all_albedo_paths]
    all_env_paths = [str(path) for path in all_env_paths]

    assert len(all_image_paths) == len(all_normal_paths)
    assert len(all_albedo_paths) == len(all_normal_paths)
    assert len(all_env_paths) == len(all_normal_paths)


    all_image_paths.sort()
    all_normal_paths.sort()
    all_albedo_paths.sort()
    all_env_paths.sort()

    image_count = len(all_image_paths)
    num_samples = image_count

    image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
    normal_paths_ds = tf.data.Dataset.from_tensor_slices(all_normal_paths)
    albedo_paths_ds = tf.data.Dataset.from_tensor_slices(all_albedo_paths)

    def load_and_preprocess_image(path):
        image = load_rgb(path)
        resize_mode = (image.shape[0] != image_shape[1] and image.shape[1] != image_shape[0])
        if (resize_mode):
            image = cv2.resize(image, (image_shape[1], image_shape[0]))
        image = image.astype(float) / 255.0  # normalize to [0,1] range
        return tf.constant(image, dtype=tf.float32)

    def load_and_preprocess_alpha(path):
        image = load_rgba(path)
        image = image.astype(float) / 255.0  # normalize to [0,1] range
        alpha = image[:,:,3]
        return tf.constant(alpha, dtype=tf.float32)

    def load_and_resize_envMap(path):
        envMap = load_rgb(path, -1)
        envMap = cv2.resize(envMap, (envMap_shape[1], envMap_shape[0]), interpolation=cv2.INTER_LINEAR)
        return tf.constant(envMap)

    def mapping_func(envMap):
        return envMap

    def create_mask(alpha):
        near_zero_tensor = tf.constant(0.01, dtype=tf.float32, shape=(image_shape[0], image_shape[1]))
        masked_a = tf.where(tf.greater(alpha, near_zero_tensor), tf.ones_like(alpha), tf.zeros_like(alpha))
        masked_a = tf.reshape(masked_a, [image_shape[0], image_shape[1], 1])
        masked_a = tf.tile(masked_a, [1, 1, 3])
        return masked_a

    all_appearances = [load_and_preprocess_image(path) for path in all_image_paths]
    train_image_ds = tf.data.Dataset.from_tensor_slices(all_appearances)

    all_normals = [load_and_preprocess_image(path) for path in all_normal_paths]
    train_normal_ds = tf.data.Dataset.from_tensor_slices(all_normals)

    all_envMaps = [load_and_resize_envMap(path) for path in all_env_paths]
    envmaps_ds = tf.data.Dataset.from_tensor_slices(all_envMaps)

    all_albedos = [load_and_preprocess_image(path) for path in all_albedo_paths]
    gt_albedo_ds = tf.data.Dataset.from_tensor_slices(all_albedos)

    train_image_ds = train_image_ds.map(mapping_func)
    train_normal_ds = train_normal_ds.map(mapping_func)
    gt_albedo_ds = gt_albedo_ds.map(mapping_func)
    gt_envmap_ds = envmaps_ds.map(mapping_func)

    alphas = [load_and_preprocess_alpha(path) for path in all_image_paths]
    alpha_ds = tf.data.Dataset.from_tensor_slices(alphas)
    mask_train_ds = alpha_ds.map(create_mask)

    result_dataset = tf.data.Dataset.zip((train_image_ds, train_image_ds, train_normal_ds,
                                          mask_train_ds, gt_albedo_ds, gt_envmap_ds))

    return (result_dataset, num_samples)

def load_envMaps(data_folder_name, num_images, input_shape):
    data_root = pathlib.Path(data_folder_name)
    all_image_paths = list(data_root.glob('*.hdr'))
    all_image_paths = [str(path) for path in all_image_paths]
    res = np.zeros([num_images, input_shape[0],input_shape[1],input_shape[2]])
    counter = 0
    for path in all_image_paths:
        envMap = load_rgb(path,-1)
        envMap = cv2.resize(envMap, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        res[counter, :,:,:] = envMap[:,:,:]
        counter += 1
    return res

def load_ground_truth_data(data_folder_name, resize_images, image_shape):
    data_root = pathlib.Path(data_folder_name)
    all_image_paths = list(data_root.glob('*Appearance_UV.png'))
    all_image_paths = [str(path) for path in all_image_paths]

    all_image_paths.sort()

    image_count = len(all_image_paths)
    num_train = image_count

    image_paths_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)

    def load_and_preprocess_image(path):
        image = tf.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        if (resize_images):
            image = tf.image.resize_images(image, [image_shape[0], image_shape[1]])
        image /= 255.0  # normalize to [0,1] range
        return image

    train_image_ds = image_paths_ds.map(load_and_preprocess_image)

    train_dataset = tf.data.Dataset.zip(train_image_ds)

    return (train_dataset, num_train)

def plot_data(x, mask):
    plt.figure(figsize=(8, 8))
    plt.gray()
    for n, image in enumerate(x.take(4)):
        plt.subplot(2, 2, n + 1)
        if mask:
            plt.imshow(image[2])
        else:
            plt.imshow(image[0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_data_bis(x, ind):
    plt.figure(figsize=(8, 8))
    plt.gray()
    for n, image in enumerate(x):
        plt.subplot(2, 2, n + 1)
        im_to_show = image[ind]
        plt.imshow(im_to_show[0])
        plt.grid(False)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plot_data_samples(data_set, indices):
    for n, sample in enumerate(data_set.take(5)):
        plt.figure(figsize=(8, 8))
        plt.gray()
        for index in indices:
            plt.subplot(2, 2, index + 1)
            im_to_show = sample[index]
            plt.imshow(im_to_show)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
        plt.show()

def plot_data_batches(data_set, indices, num_batches):
    for n, batch in enumerate(data_set.take(num_batches)):
        plt.figure(figsize=(8, 8))
        plt.gray()
        counter = 0
        for index in indices:
            sample = batch[index]
            plt.subplot(2, 2, counter + 1)
            im_to_show = sample[0]
            plt.imshow(im_to_show)
            plt.grid(False)
            plt.xticks([])
            plt.yticks([])
            counter+=1
        plt.show()
