import logging

import tensorflow as tf
from tensorflow.keras import layers


def batched_segment_op(data,
                       segment_ids,
                       num_segments,
                       data_mask=None,
                       reduction='sum'):
    """Flattens data and segment_ids containing a batch dimension for
    tf.math.segment* operations. Includes support for masking.

    Arguments:
        data: tensor of shape [B, L, F], where B is the batch size, L is the
            length, and F is a feature dimension
        segment_ids: tensor of shape [B, L] containing up to N segments
        num_segments: N, integer
        data_mask: boolean tensor of shape [B, L] masking the input data
        reduction: string for specific tf.math.unsorted_segment_* function

    """

    if data_mask is None:
        data_mask = tf.ones(tf.shape(data)[:-1], dtype=tf.bool)

    # Prior to flattening, offset rows of segment_ids to preserve batches
    batch_size = tf.shape(data, out_type=segment_ids.dtype)[0]
    offsets = tf.range(batch_size, dtype=segment_ids.dtype) * num_segments
    ids_offset = segment_ids + tf.expand_dims(offsets, 1)

    # Mask and flatten the data and segment_ids
    flat_data = tf.boolean_mask(data, data_mask)
    flat_ids = tf.boolean_mask(ids_offset, data_mask)

    reduction = getattr(tf.math, f'unsorted_segment_{reduction}')

    # Perform the segment operation on the flattened vectors, and reshape the
    # result
    reduced_data = reduction(flat_data, flat_ids, num_segments * batch_size)
    return tf.reshape(reduced_data, [batch_size, num_segments, data.shape[-1]])


class Slice(layers.Layer):
    def __init__(self, slice_obj, *args, **kwargs):
        super(Slice, self).__init__(*args, **kwargs)
        self.slice_obj = slice_obj
        self.supports_masking = True

    def call(self, inputs, mask=None):
        return inputs[self.slice_obj]

    def get_config(self):
        return {'slice_obj': str(self.slice_obj)}

    @classmethod
    def from_config(cls, config):
        config['slice_obj'] = eval(config['slice_obj'])
        return cls(**config)


class Gather(layers.Layer):
    def call(self, inputs, mask=None):
        reference, indices = inputs
        return tf.gather(reference, indices, batch_dims=1)


class Reduce(layers.Layer):
    def __init__(self, reduction='sum', *args, **kwargs):
        super(Reduce, self).__init__(*args, **kwargs)
        self.reduction = reduction

    def _parse_inputs_and_mask(self, inputs, mask=None):
        data, segment_ids, target = inputs

        # Handle missing masks
        if mask is not None:
            data_mask = mask[0]
        else:
            data_mask = None

        return data, segment_ids, target, data_mask

    def call(self, inputs, mask=None):
        data, segment_ids, target, data_mask = self._parse_inputs_and_mask(
            inputs, mask)
        num_segments = tf.shape(target, out_type=segment_ids.dtype)[1]
        return batched_segment_op(
            data,
            segment_ids,
            num_segments,
            data_mask=data_mask,
            reduction=self.reduction)

    def get_config(self):
        return {'reduction': self.reduction}


class ConcatDense(layers.Layer):
    """ Layer to combine the concatenation and two dense layers. Just useful as a common operation in the graph
    layers """

    def __init__(self, **kwargs):
        super(ConcatDense, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        num_features = input_shape[0][-1]
        self.concat = layers.Concatenate()
        self.dense1 = layers.Dense(2 * num_features, activation='relu')
        self.dense2 = layers.Dense(num_features)

    def call(self, inputs, mask=None):
        output = self.concat(inputs)
        output = self.dense1(output)
        output = self.dense2(output)
        return output

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return tf.math.reduce_all(tf.stack(mask), axis=0)


class Tile(layers.Layer):
    def __init__(self, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        global_state, target = inputs
        target_shape = tf.shape(target)[1]  # number of edges or nodes
        expanded = tf.expand_dims(global_state, 1)
        return tf.tile(expanded, tf.stack([1, target_shape, 1]))

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return mask[1]
