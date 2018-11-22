import numpy as np
import pandas as pd

import tensorflow as tf
print(tf.__version__)

import keras
print(keras.__version__)

data = pd.read_csv('data/delaney.csv')

data.head()

test = data.sample(frac=0.2, random_state=0)
train = data[~data.index.isin(test.index)]

from sklearn.preprocessing import StandardScaler

y_train = train['measured log solubility in mols per litre'].values.reshape(-1, 1)
y_test = test['measured log solubility in mols per litre'].values.reshape(-1, 1)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train)
y_test_scaled = y_scaler.transform(y_test)

from nfp.preprocessing import SmilesPreprocessor, GraphSequence

preprocessor = SmilesPreprocessor(explicit_hs=False, feature_set='v2')

inputs_train = preprocessor.fit(train.smiles)
inputs_test = preprocessor.predict(test.smiles)

batch_size = 32
train_generator = GraphSequence(inputs_train, y_train_scaled, 32)
test_generator = GraphSequence(inputs_test, y_test_scaled, 32)

import warnings

# Define Keras model
import keras
import keras.backend as K

import tensorflow as tf

from keras.layers import (
    Input, Embedding, Dense, BatchNormalization, Reshape, Lambda, Activation)

from keras.models import Model
from keras.engine import Layer

from nfp.layers import MessageLayer, GRUStep, GraphOutput, Embedding2D, Squeeze
from nfp.models import GraphModel

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=False))
K.set_session(sess)

with tf.device('/gpu:0'):

    # Raw (integer) graph inputs
    node_graph_indices = Input(shape=(1,), name='node_graph_indices', dtype='int32')
    atom_types = Input(shape=(1,), name='atom', dtype='int32')
    bond_types = Input(shape=(1,), name='bond', dtype='int32')
    connectivity = Input(shape=(2,), name='connectivity', dtype='int32')

    squeeze = Squeeze()

    snode_graph_indices = squeeze(node_graph_indices)
    satom_types = squeeze(atom_types)
    sbond_types = squeeze(bond_types)

    # Initialize RNN and MessageLayer instances
    atom_features = 20

    # Initialize the atom states
    atom_state = Embedding(
        preprocessor.atom_classes,
        atom_features, name='atom_embedding')(satom_types)

    # Initialize the bond states
    bond_matrix = Embedding2D(
        preprocessor.bond_classes,
        atom_features, name='bond_embedding')(sbond_types)

    atom_rnn_layer = GRUStep(atom_features)
    message_layer = MessageLayer(reducer='sum')

    message_steps = 3
    # Perform the message passing
    for _ in range(message_steps):

        # Get the message updates to each atom
        message = message_layer([atom_state, bond_matrix, connectivity])

        # Update memory and atom states
        atom_state = atom_rnn_layer([message, atom_state])
        
    atom_fingerprint = Dense(64, activation='relu')(atom_state)
    mol_fingerprint = GraphOutput(reducer='sum')([snode_graph_indices, atom_fingerprint])
    mol_fingerprint = BatchNormalization()(mol_fingerprint)

    out = Dense(1)(mol_fingerprint)
    model = GraphModel([node_graph_indices, atom_types, bond_types, connectivity], [out])
    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')
    model.summary()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hist = model.fit_generator(train_generator, validation_data=test_generator, epochs=50, verbose=2)
