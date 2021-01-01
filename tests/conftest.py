import pytest
import tensorflow as tf

import nfp


@pytest.fixture(scope='module')
def smiles_inputs():
    preprocessor = nfp.SmilesPreprocessor()
    dataset = tf.data.Dataset.from_generator(
        lambda: (preprocessor.construct_feature_matrices(smiles, train=True)
                 for smiles in ['CC', 'CCC', 'C(C)C', 'C']),
        output_types=preprocessor.output_types,
        output_shapes=preprocessor.output_shapes) \
        .padded_batch(batch_size=4,
                      padded_shapes=preprocessor.padded_shapes(),
                      padding_values=preprocessor.padding_values)

    return preprocessor, list(dataset.take(1))[0]
