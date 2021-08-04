import pytest
import tensorflow as tf

import nfp


@pytest.fixture(scope='module')
def smiles_inputs():
    preprocessor = nfp.SmilesPreprocessor()
    dataset = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
                 for smiles in ['CC', 'CCC', 'C1CC1', 'C']),
        output_signature=preprocessor.output_signature) \
        .padded_batch(batch_size=4)

    return preprocessor, list(dataset.take(1))[0]


@pytest.fixture(scope='module')
def inputs_no_padding(smiles_inputs):
    preprocessor, inputs = smiles_inputs

    dataset = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
                 for smiles in ['CC', 'CCC', 'C(C)C', 'C']),
        output_signature=preprocessor.output_signature) \
        .padded_batch(batch_size=4)

    return list(dataset.take(1))[0]


@pytest.fixture(scope='module')
def inputs_with_padding(smiles_inputs):
    preprocessor, inputs = smiles_inputs

    dataset = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
                 for smiles in ['CC', 'CCC', 'C(C)C', 'C']),
        output_signature=preprocessor.output_signature) \
        .padded_batch(batch_size=4,
                      padded_shapes={'atom': (20,), 'bond': (40,), 'connectivity': (40, 2)})

    return list(dataset.take(1))[0]
