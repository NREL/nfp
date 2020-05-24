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

    def call(self, inputs):
        return inputs[self.slice_obj]

    def compute_mask(self, inputs, mask):
        return mask

    def get_config(self):
        return {'slice_obj': str(self.slice_obj)}
    
    @classmethod
    def from_config(cls, config):
        config['slice_obj'] = eval(config['slice_obj'])
        return cls(**config)


class Gather(layers.Layer):
    def call(self, inputs):
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
