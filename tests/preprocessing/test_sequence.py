import pandas as pd
import numpy as np

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nfp.preprocessing import SmilesPreprocessor, GraphSequence

def test_sequence(get_2d_data):
    
    inputs, y = get_2d_data

    assert len(GraphSequence(inputs, y, batch_size=5, final_batch=True)) == 11
    assert len(GraphSequence(inputs, y, batch_size=5, final_batch=False)) == 11

    assert len(GraphSequence(inputs, y, batch_size=10, final_batch=True)) == 6
    assert len(GraphSequence(inputs, y, batch_size=10, final_batch=False)) == 5

    seq = GraphSequence(inputs, y, batch_size=10, final_batch=False)
    for batch in seq:
        assert batch[0]['node_graph_indices'][-1] == 9

    np.testing.assert_allclose(
        GraphSequence(inputs, y, batch_size=3, shuffle=False)[0][0]['atom'][:10],
        GraphSequence(inputs, y=None, batch_size=5, shuffle=False)[0]['atom'][:10])

    seq1 = GraphSequence(inputs, y, batch_size=3, shuffle=True)
    seq2 = GraphSequence(inputs, y, batch_size=3, shuffle=True)
    seq2.on_epoch_end()

    assert ~np.allclose(seq1[0][1], seq2[0][1])


    seq1 = GraphSequence(inputs, y=None, batch_size=1, shuffle=False)
    seq2 = GraphSequence(inputs, y=None, batch_size=3, shuffle=False)
    
    np.testing.assert_allclose(
        np.concatenate([seq1[0]['atom'], seq1[1]['atom'], seq1[2]['atom']]),
        seq2[0]['atom'])

    assert len(seq2[0]['atom']) == (seq2[0]['connectivity'].max() + 1)
    assert seq2[0]['connectivity'].min() == 0

    num_atoms = sum((len(item['atom']) for item in inputs))
    num_atoms_except_last = sum((len(item['atom']) for item in inputs[:-1]))

    all_batches = seq1[:]
    assert num_atoms == len(all_batches['atom'])

    all_batches_except_last = seq2[:-1]
    assert num_atoms_except_last == len(all_batches_except_last['atom'])
