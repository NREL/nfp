import pandas as pd
import numpy as np

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from nfp.preprocessing import SmilesPreprocessor, GraphSequence


def test_sequence(get_2d_data):

    # Unpack the test data (from a module-level fixture)
    inputs, y = get_2d_data

    # Test counting the number of batches
    assert len(GraphSequence(inputs, y, batch_size=5, final_batch=True)) == 11
    assert len(GraphSequence(inputs, y, batch_size=5, final_batch=False)) == 11

    assert len(GraphSequence(inputs, y, batch_size=10, final_batch=True)) == 6
    assert len(GraphSequence(inputs, y, batch_size=10, final_batch=False)) == 5

    # Test that atoms are assigned to the proper molecule
    seq = GraphSequence(inputs, y, batch_size=10, final_batch=False)
    for batch in seq:
        assert batch[0]['node_graph_indices'][-1] == 9
        assert batch[0]['bond_graph_indices'][-1] == 9

        assert len(batch[0]['node_graph_indices']) == len(batch[0]['atom'])
        assert len(batch[0]['bond_graph_indices']) == len(batch[0]['bond'])

    # Test results are identical w/ and w/o output data
    np.testing.assert_allclose(
        GraphSequence(inputs, y, batch_size=3, shuffle=False)[0][0]['atom'][:10],
        GraphSequence(inputs, y=None, batch_size=5, shuffle=False)[0]['atom'][:10])

    # Verify that shuffling actually changes the order of entries
    seq1 = GraphSequence(inputs, y, batch_size=3, shuffle=True)
    seq2 = GraphSequence(inputs, y, batch_size=3, shuffle=True)
    seq2.on_epoch_end()

    assert ~np.allclose(seq1[0][1], seq2[0][1])

    # Verify that shuffling with an offset yields the expected result
    #   Offset by the equivalent of one batch
    seq3 = GraphSequence(inputs, y, batch_size=3, shuffle_offset=3)
    seq3.on_epoch_end()

    assert np.allclose(seq2[0][1], seq3[1][1])

    # Check that molecules are concatenated in the proper order and connectivity are updated
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

    # Test output is a 2D array
    y_list = [y, y + 1]
    seq_list = GraphSequence(inputs, y_list, batch_size=5, final_batch=True)
    assert len(seq_list[0][1][0]) == 5
    assert np.allclose(seq_list[0][1][0], seq_list[0][1][1] - 1)

    # Test outputs as a dictionary
    y_dict = {'y1': y, 'y2': y + 1}
    seq_dict = GraphSequence(inputs, y_dict, batch_size=5, final_batch=True)
    assert len(seq_dict[0][1]['y1']) == 5
    assert np.allclose(seq_dict[0][1]['y1'], seq_dict[0][1]['y2'] - 1)
