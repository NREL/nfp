from nfp.frameworks import tf

assert tf, "Tensorflow 2.x required for GraphLayers"
tf_layers = tf.keras.layers


class RBFExpansion(tf_layers.Layer):
    def __init__(
        self, dimension=128, init_gap=10, init_max_distance=7, trainable=False
    ):
        """Layer to calculate radial basis function 'embeddings' for a continuous input
        variable. The width and location of each bin can be optionally trained.
        Essentially equivalent to a 1-hot embedding for a continuous variable.

        Parameters
        ----------
        dimension: The total number of distance bins
        init_gap: The initial width of each gaussian distribution
        init_max_distance: the initial maximum value of the continuous variable
        trainable: Whether the centers and gap parameters should be added as trainable
            NN parameters.
        """
        super(RBFExpansion, self).__init__()
        self.init_gap = init_gap
        self.init_max_distance = init_max_distance
        self.dimension = dimension
        self.trainable = trainable

    def build(self, input_shape):
        self.centers = tf.Variable(
            name="centers",
            initial_value=tf.range(
                0, self.init_max_distance, delta=self.init_max_distance / self.dimension
            ),
            trainable=self.trainable,
            dtype=tf.float32,
        )

        self.gap = tf.Variable(
            name="gap",
            initial_value=tf.constant(self.init_gap, dtype=tf.float32),
            trainable=self.trainable,
            dtype=tf.float32,
        )

    def call(self, inputs, **kwargs):
        distances = tf.where(
            tf.math.is_nan(inputs), tf.zeros_like(inputs, dtype=inputs.dtype), inputs
        )
        offset = tf.expand_dims(distances, -1) - tf.cast(self.centers, inputs.dtype)
        logits = -self.gap * offset ** 2
        return tf.exp(logits)

    def get_config(self):
        return {
            "init_gap": self.init_gap,
            "init_max_distance": self.init_max_distance,
            "dimension": self.dimension,
            "trainable": self.trainable,
        }


class ConcatDense(tf_layers.Layer):
    """Layer to combine the concatenation and two dense layers. Just useful as a common
    operation in the graph layers"""

    def __init__(self, **kwargs):
        super(ConcatDense, self).__init__(**kwargs)

    def build(self, input_shape):
        num_features = input_shape[0][-1]
        self.concat = tf_layers.Concatenate()
        self.dense1 = tf_layers.Dense(2 * num_features, activation="relu")
        self.dense2 = tf_layers.Dense(num_features)

    def call(self, inputs, mask=None, **kwargs):
        output = self.concat(inputs)
        output = self.dense1(output)
        output = self.dense2(output)
        return output
