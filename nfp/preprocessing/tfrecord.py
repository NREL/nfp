import numpy as np
import tensorflow as tf


# Code from https://www.tensorflow.org/tutorials/load_data/tfrecord

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_value(value):
    if type(value) == np.ndarray:
        return _bytes_feature(tf.io.serialize_tensor(value))
    elif type(value) == int:
        return _int64_feature(value)
    elif type(value) == float:
        return _float_feature(value)
    else:
        raise TypeError(f"Didn't recognize type {type(value)}")
