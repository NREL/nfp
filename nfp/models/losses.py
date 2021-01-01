""" Keras loss functions will choke on NaN inputs, which we'll often have for
unspecified inputs. These are just a couple simple loss functions that mask NaN
values in both the test and predicted tensors when computing the cost. """

import tensorflow as tf


def masked_mean_squared_error(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    y_true_mask = tf.boolean_mask(tensor=y_true, mask=mask)
    y_pred_mask = tf.boolean_mask(tensor=y_pred, mask=mask)
    return tf.math.reduce_mean(tf.math.square(y_pred_mask - y_true_mask))


def masked_mean_absolute_error(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    y_true_mask = tf.boolean_mask(tensor=y_true, mask=mask)
    y_pred_mask = tf.boolean_mask(tensor=y_pred, mask=mask)
    return tf.math.reduce_mean(tf.math.abs(y_pred_mask - y_true_mask))


def masked_log_cosh(y_true, y_pred):
    mask = tf.math.is_finite(y_true)
    y_true_mask = tf.boolean_mask(tensor=y_true, mask=mask)
    y_pred_mask = tf.boolean_mask(tensor=y_pred, mask=mask)
    return tf.keras.losses.logcosh(y_pred_mask, y_true_mask)
