# Demonstrates interoperability of the nfp preprocessing methods with the spektral GNN
# library (https://graphneural.network/). Specifically follows the `BatchLoader` example
# for QM9 prediction shown here, https://github.com/danielegrattarola/spektral/blob/
# master/examples/graph_prediction/qm9_ecc_batch.py

import pandas as pd  # noqa: F401
import tensorflow as tf
from tensorflow.keras import layers

from spektral import layers as slayers
from spektral.data import DisjointLoader
from ysi_data import dataset_te, dataset_tr, preprocessor

batch_size = 32
epochs = 5

loader_tr = DisjointLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_te = DisjointLoader(dataset_te, batch_size=batch_size)


class Net(tf.keras.Model):
    def __init__(self, atom_classes, bond_classes):
        super().__init__()

        self.atom_embedding = layers.Embedding(atom_classes, 32)
        self.atom_bias = layers.Embedding(atom_classes, 1)

        self.bond_embedding = layers.Embedding(bond_classes, 32)
        self.masking = slayers.GraphMasking()
        self.conv1 = slayers.ECCConv(32, activation="relu")
        self.conv2 = slayers.ECCConv(32, activation="relu")
        self.dense = layers.Dense(
            1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1e-5)
        )
        self.global_pool = slayers.GlobalSumPool()

    def call(self, inputs):
        x_in, a, e, i = inputs

        x = self.atom_embedding(tf.squeeze(x_in, axis=[-1]))
        e = self.bond_embedding(tf.squeeze(e, axis=[-1]))

        x = self.conv1([x, a, e])
        x += self.conv2([x, a, e])

        output = self.atom_bias(tf.squeeze(x_in, axis=[-1]))
        output += self.dense(x)
        output = self.global_pool([output, i])

        return output


model = Net(preprocessor.atom_classes, preprocessor.bond_classes)
optimizer = tf.keras.optimizers.Adam(1e-3)
model.compile(optimizer=optimizer, loss="mae")

model.fit(
    loader_tr.load(),
    validation_data=loader_te.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_steps=loader_te.steps_per_epoch,
    epochs=5,
)
