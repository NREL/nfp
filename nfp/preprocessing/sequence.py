import numpy as np

from keras.utils import Sequence


class GraphSequence(Sequence):
    """A keras.Sequence generator to be passed to model.fit_generator. (or
        any other *_generator method.) Returns (inputs, y) tuples where
        molecule feature matrices have been stitched together. Offsets the
        connectivity matrices such that atoms are indexed appropriately.
    """
    
    def __init__(self, inputs, y=None, batch_size=1, shuffle=True,
                 final_batch=True, seed=1, shuffle_offset=0):
        """

        Args:
            inputs ([dict]): List of dictionaries describing each molecule
            y (list, dict, None): Output properties. Either a 1- or 2-d list, or a dictionary
                of lists, or "None" if there are no outputs
            batch_size (int): Number of molecules per batch
            shuffle (bool): Whether to shuffle the input data
            final_batch (bool): Whether to include the final, incomplete batch
            seed (int): Initial random seed for the data shuffler
            shuffle_offset (int): How much to offset the indices when shuffling the data.
                Used during distributed model training to ensure no data overlap between replicas
        """
        self._inputs = inputs
        self._y = y
        self._input_keys = list(inputs[0].keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.final_batch = final_batch
        self._random = np.random.RandomState(seed)
        self.shuffle_offset = shuffle_offset

    def __len__(self):
        """ Total number of batches """
        if self.final_batch:
            return int(np.ceil(len(self._inputs) / float(self.batch_size)))
        else:
            return int(np.floor(len(self._inputs) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            # Shuffle the indices
            indices = np.arange(0, len(self._inputs))
            self._random.shuffle(indices)

            # If needed, shift the data to create an offset
            if self.shuffle_offset != 0:
                indices = np.roll(indices, self.shuffle_offset)

            # Arrange the list according to the shuffle
            self._inputs = [self._inputs[i] for i in indices]
            if self._y is not None:
                self._y = self.index_y(indices)

    def __getitem__(self, idx):
        """ Calculate the feature matrices for a whole batch (with index `i` <
        self.__len__). This involves adding offsets to the indices for each
        atom in the connectivity matrix; such that atoms and bonds in later
        molecules still refer to the correct atoms.

        """

        if isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            batch_indexes = np.concatenate(
                [i * self.batch_size + np.arange(0, self.batch_size)
                 for i in indices])

        else:
            batch_indexes = idx * self.batch_size + np.arange(0, self.batch_size)

        batch_indexes = batch_indexes[batch_indexes < len(self._inputs)]

        batch_data = {
            key: self._concat([self._inputs[i][key] for i in batch_indexes])
            for key in self._input_keys}
    
        # Offset the connectivity matrix to account for the multiple graphs per
        # batch
        offset = _compute_stacked_offsets(
            batch_data['n_atom'], batch_data['n_bond'])

        batch_data['connectivity'] += offset[:, np.newaxis]
        
        # Compute graph indices with shape (n_atom,) that indicate to which
        # molecule each atom belongs.
        n_graphs = len(batch_indexes)
        batch_data['node_graph_indices'] = np.repeat(
            np.arange(n_graphs), batch_data['n_atom'])
        batch_data['bond_graph_indices'] = np.repeat(
            np.arange(n_graphs), batch_data['n_bond'])

        batch_data = self.process_data(batch_data)

        # Keras takes to options, one (x, y) pairs, or just (x,) pairs if we're
        # doing predictions. Here, if we've specified a y matrix, we return the
        # x,y pairs for training, otherwise just return the x data.
        if self._y is not None:
            return batch_data, self.index_y(batch_indexes)
        else:
            return batch_data

    def index_y(self, indices):
        """ the output field can take a bunch of different formats """

        if isinstance(self._y, list):
            return [yi[indices] for yi in self._y]

        elif isinstance(self._y, dict):
            return {key: val[indices] for key, val in self._y.items()}

        else:
            return np.asarray(self._y)[indices]

    def process_data(self, batch_data):
        """ function to add additional processing to batch data before returning """

        # These aren't used currently, so I pop them. But we might need them at
        # a later time.
        # del batch_data['n_atom']
        # del batch_data['n_bond']
        
        return batch_data

    def _concat(self, to_stack):
        """ function to stack (or concatenate) depending on dimensions """

        if np.asarray(to_stack[0]).ndim >= 2:
            return np.concatenate(to_stack)
        
        else:
            return np.hstack(to_stack)


def _compute_stacked_offsets(sizes, repeats):
    """ Computes offsets to add to indices of stacked np arrays.
    When a set of np arrays are stacked, the indices of those from the second on
    must be offset in order to be able to index into the stacked np array. This
    computes those offsets.

    Args:
        sizes: A 1D sequence of np arrays of the sizes per graph.
        repeats: A 1D sequence of np arrays of the number of repeats per graph.
    Returns:
        The index offset per graph.
    """
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)
