from collections import Counter

import nfp.preprocessing.crystal_preprocessor
import numpy as np


def test_pymatgen_preprocessor(structure_inputs):
    preprocessor = nfp.preprocessing.crystal_preprocessor.PymatgenPreprocessor()
    inputs = preprocessor(structure_inputs[0], train=True)

    for val in Counter(inputs["connectivity"][:, 0]).values():
        # Assert each site has the right number of neighbors
        assert val == preprocessor.num_neighbors

    assert len(inputs["site"]) == structure_inputs[0].num_sites
    assert (
        len(inputs["distance"])
        == structure_inputs[0].num_sites * preprocessor.num_neighbors
    )


def test_padding(structure_inputs):
    preprocessor = nfp.preprocessing.crystal_preprocessor.PymatgenPreprocessor()
    inputs = preprocessor(structure_inputs[0], train=True, max_num_edges=100)
    assert np.isnan(inputs["distance"][-1])
