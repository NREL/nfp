import tensorflow as tf
from tensorflow.keras import layers as tf_layers


class RBFExpansion(tf_layers.Layer):
    def __init__(self,
                 dimension=128,
                 init_gap=10,
                 init_max_distance=7,
                 trainable=False):
        """ Layer to calculate radial basis function 'embeddings' for a continuous input variable. The width and
        location of each bin can be optionally trained. Essentially equivalent to a 1-hot embedding for a continuous
        variable.

        Parameters
        ----------
        dimension: The total number of distance bins
        init_gap: The initial width of each gaussian distribution
        init_max_distance: the initial maximum value of the continuous variable
        trainable: Whether the centers and gap parameters should be added as trainable NN parameters.
        """
        super(RBFExpansion, self).__init__()
        self.init_gap = init_gap
        self.init_max_distance = init_max_distance
        self.dimension = dimension
        self.trainable = trainable

    def build(self, input_shape):
        self.centers = tf.Variable(
            name='centers',
            initial_value=tf.range(
                0, self.init_max_distance,
                delta=self.init_max_distance / self.dimension),
            trainable=self.trainable,
            dtype=tf.float32)

        self.gap = tf.Variable(name='gap',
                               initial_value=tf.constant(self.init_gap,
                                                         dtype=tf.float32),
                               trainable=self.trainable,
                               dtype=tf.float32)

    def call(self, inputs, **kwargs):
        distances = tf.where(tf.math.is_nan(inputs),
                             tf.zeros_like(inputs, dtype=inputs.dtype), inputs)
        offset = tf.expand_dims(distances, -1) - tf.cast(
            self.centers, inputs.dtype)
        logits = -self.gap * offset ** 2
        return tf.exp(logits)

    def compute_mask(self, inputs, mask=None):
        return tf.logical_not(tf.math.is_nan(inputs))

    def get_config(self):
        return {
            'init_gap': self.init_gap,
            'init_max_distance': self.init_max_distance,
            'dimension': self.dimension,
            'trainable': self.trainable
        }


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


# class Slice(tf_layers.Layer):
#     def __init__(self, slice_obj, *args, **kwargs):
#         super(Slice, self).__init__(*args, **kwargs)
#         self.slice_obj = slice_obj
#         self.supports_masking = True
#
#     def call(self, inputs, mask=None):
#         return inputs[self.slice_obj]
#
#     def get_config(self):
#         return {'slice_obj': str(self.slice_obj)}
#
#     @classmethod
#     def from_config(cls, config):
#         config['slice_obj'] = eval(config['slice_obj'])
#         return cls(**config)


class Gather(tf_layers.Layer):
    def call(self, inputs, mask=None, **kwargs):
        reference, indices = inputs
        return tf.gather(reference, indices, batch_dims=1)


class Reduce(tf_layers.Layer):
    def __init__(self, reduction='sum', *args, **kwargs):
        super(Reduce, self).__init__(*args, **kwargs)
        self.reduction = reduction

    def compute_output_shape(self, input_shape):
        data_shape, _, target_shape = input_shape
        return [data_shape[0], target_shape[1], data_shape[-1]]

    def call(self, inputs, mask=None, **kwargs):
        data, segment_ids, target, data_mask = self._parse_inputs_and_mask(
            inputs, mask)
        num_segments = tf.shape(target, out_type=segment_ids.dtype)[1]
        return batched_segment_op(data,
                                  segment_ids,
                                  num_segments,
                                  data_mask=data_mask,
                                  reduction=self.reduction)

    def get_config(self):
        return {'reduction': self.reduction}

    @staticmethod
    def _parse_inputs_and_mask(inputs, mask=None):
        data, segment_ids, target = inputs

        # Handle missing masks
        if mask is not None:
            data_mask = mask[0]
        else:
            data_mask = None

        return data, segment_ids, target, data_mask


class ConcatDense(tf_layers.Layer):
    """ Layer to combine the concatenation and two dense layers. Just useful as a common operation in the graph
    layers """

    def __init__(self, **kwargs):
        super(ConcatDense, self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        num_features = input_shape[0][-1]
        self.concat = tf_layers.Concatenate()
        self.dense1 = tf_layers.Dense(2 * num_features, activation='relu')
        self.dense2 = tf_layers.Dense(num_features)

    def call(self, inputs, mask=None, **kwargs):
        output = self.concat(inputs)
        output = self.dense1(output)
        output = self.dense2(output)
        return output

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return tf.math.reduce_all(tf.stack(mask), axis=0)


class Tile(tf_layers.Layer):
    def __init__(self, **kwargs):
        super(Tile, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        global_state, target = inputs
        target_shape = tf.shape(target)[1]  # number of edges or nodes
        expanded = tf.expand_dims(global_state, 1)
        return tf.tile(expanded, tf.stack([1, target_shape, 1]))

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        else:
            return mask[1]
