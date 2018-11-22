import sys

model_name = 'b3lyp_2D_noatom_bn_fixed'
print(model_name)

import os
import numpy as np
import pandas as pd
import pickle

import keras
from keras.layers import (Add, Input, Dense, BatchNormalization,
                          Activation, Dropout, Embedding, Lambda)
from keras.models import Model
from keras.callbacks import ModelCheckpoint, CSVLogger

from nfp.preprocessing import SmilesPreprocessor, RobustNanScaler, GraphSequence
from nfp.models import masked_mean_squared_error

from nfp.layers import (MessageLayer, GRUStep, ReduceAtomToMol, Embedding2D, Squeeze)
from nfp.models import GraphModel


# Load the input data
train = pd.read_csv('smiles_data/train.csv.gz', index_col=0).sample(frac=1.)
valid = pd.read_csv('smiles_data/valid.csv.gz', index_col=0).sample(frac=1.)

# Transform SMILES strings into X matrix
preprocessor = SmilesPreprocessor(explicit_hs=True)

train_inputs = preprocessor.fit(train.smile)
valid_inputs = preprocessor.predict(valid.smile)

# Rescale Y matrix
y_train_raw = train.set_index('smile').values
y_valid_raw = valid.set_index('smile').values

y_scaler = RobustNanScaler()
y_train_scaled = y_scaler.fit_transform(y_train_raw)
y_valid_scaled = y_scaler.transform(y_valid_raw)

batch_size = 100
train_generator = GraphSequence(train_inputs, y_train_scaled, batch_size)
valid_generator = GraphSequence(valid_inputs, y_valid_scaled, batch_size)

num_output = y_train_scaled.shape[1]  # For model output


# Save the preprocessor classes
if not os.path.exists(model_name):
    os.makedirs(model_name)

with open(model_name + '/preprocessor.p', 'wb') as f:
    pickle.dump(preprocessor, f)
    
with open(model_name + '/y_scaler.p', 'wb') as f:
    pickle.dump(y_scaler, f)


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
atom_features = 128

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

# atom_state = BatchNormalization(momentum=0.9)(atom_state)
atom_fingerprint = Dense(1024, activation='sigmoid')(atom_state)
mol_out = ReduceAtomToMol(reducer='sum')([atom_fingerprint, snode_graph_indices])

X = BatchNormalization(momentum=0.9)(mol_out)
X = Dense(512, activation='relu')(X)

X = BatchNormalization(momentum=0.9)(X)
X = Dense(256, activation='relu')(X)
X = Dense(num_output)(X)
    
model = GraphModel([node_graph_indices, atom_types, bond_types, connectivity], [X])

epochs = 500
lr = 1E-3
decay = lr/epochs

model.compile(optimizer=keras.optimizers.Adam(lr=lr, decay=decay),
              loss=masked_mean_squared_error)

model.summary()
    
filepath= model_name + "/best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, save_best_only=True, period=10, verbose=1)
csv_logger = CSVLogger(model_name + '/log.csv')

hist = model.fit_generator(
	train_generator, validation_data=valid_generator, verbose=1,
	epochs=epochs, callbacks=[checkpoint, csv_logger])
