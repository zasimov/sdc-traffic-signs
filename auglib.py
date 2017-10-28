"""Augmentation =)

Use augment function:
   augment(input_dataset, **options)

"""

import functools
import random

import cv2
import numpy as np
import tensorflow as tf


def rotate3(angle_degree, image):
    """rotate3 returns rotated image

    Parameters:
      - angle_degree - angle (degree)
      - image - image tensor 3D

    """
    angle = random.randint(-angle_degree, angle_degree)
    cols, rows, _ = list(map(int, image.shape))
    rotate_matrix = cv2.getRotationMatrix2D((cols/2,rows/2), angle, 1)
    return cv2.warpAffine(image, rotate_matrix, (cols,rows))


def rotate4(angle_degree, images4d):
    # FIXME: I'm not sure it's memory-effective way
    rotated = [rotate3(angle_degree, image) for image in images4d]
    return np.array(rotated, images4d.dtype)


def tf_random_brightness_contrast(brightness_max_delta, contrast_lower, contrast_upper, image):
    """change brightness and contrast in random way

    image should be 3D tensor
    """
    image = tf.image.random_brightness(image, max_delta=brightness_max_delta)
    image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
    return image


def augment_features(images4d,
                     angle_degree=15,
                     brightness_max_delta=0.1,
                     contrast_lower=0.8, contrast_upper=1.2):
    """rotate, change brightness and contrast

    images4d - 4D tensor with images (samples.features for example)

    Function uses

    Returns augmented features

    """
    images4d = rotate4(angle_degree, images4d)

    bc = functools.partial(tf_random_brightness_contrast,
                           brightness_max_delta,
                           contrast_lower, contrast_upper)

    with tf.Session() as sess:
        images4d = sess.run(tf.map_fn(bc, images4d))

    return images4d


def augment_samples(samples, **kwargs):
    """augment samples (helpers.Samples)"""
    augmented_features = augment_features(samples.features)
    return samples.__class__(augmented_features, samples.labels)
