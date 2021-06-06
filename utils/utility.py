import numpy as np
import tensorflow as tf
import os
import time
import GPUtil
import functools
import random

def lazy_property_with_scope(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__, reuse=tf.AUTO_REUSE):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

def mask_gpu(gpu_index=None):
    gpus = GPUtil.getGPUs()

    if gpu_index is None:
        mem_frees = [gpu.memoryFree for gpu in gpus]
        gpu_index = mem_frees.index(max(mem_frees))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[gpu_index].id)

def seed_everything(seed=1126):
    os.environ['PYTHONHASHSEED']=str(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def prepare_session(mem=None):
    config = tf.ConfigProto()
    if mem is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = mem
    while True:
        try:
            sess = tf.Session(config=config)
        except:
            time.sleep(30)
            continue
        break
    return sess

def deserialize_example(serialized_example):
    feature = {
        'seqlen':tf.FixedLenFeature([],tf.int64),
        'img':tf.FixedLenFeature([], tf.string),
        'intensity':tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(serialized_example ,feature)
    seqlen = tf.cast(example['seqlen'], tf.int32)

    img_shape = [seqlen, 201, 201, 4]
    intensity_shape = [seqlen]
    
    img = tf.decode_raw(example['img'], tf.float32)
    intensity = tf.decode_raw(example['intensity'], tf.float64)
    intensity = tf.cast(intensity, tf.float32)
    img = tf.reshape(img, img_shape)
    intensity = tf.reshape(intensity, intensity_shape)

    return img, intensity, seqlen

def rotate_and_crop(img, intensity, seqlen, rotate_type=None, return_angle=False):

    intial_crop_x = img
    # intial_crop_x = tf.image.crop_to_bounding_box(img, 54, 54, 93, 93)
    two_channel_x = tf.stack([intial_crop_x[:, :, :, 0], intial_crop_x[:, :, :, 3]], -1)

    if rotate_type == 'single':
        angles = tf.random_uniform([tf.shape(two_channel_x)[0]], maxval=360)
    elif rotate_type == 'series':
        angles = tf.ones([tf.shape(two_channel_x)[0]]) * tf.random_uniform([1], maxval=360)
    else:
        angles = tf.zeros([tf.shape(two_channel_x)[0]])

    rotated_x = tf.contrib.image.rotate(two_channel_x ,angles=angles)

    center_x = tf.image.crop_to_bounding_box(rotated_x, 14, 14, 64, 64)

    if return_angle:
        center_x = tf.reshape(center_x, [tf.shape(center_x)[0], -1]) # shape = [T, W * H * C]
        angles = tf.expand_dims(angles, axis=-1)
        center_x = tf.concat([center_x, angles], axis=-1) # shape = [T, W * H * C + 1]

    return center_x, intensity, seqlen
