class Tokenizer(object):
    """ A class to turn arbitrary inputs into integer classes. """

    def __init__(self):
        # the default class for an unseen entry during test-time
        self._data = {'unk': 1}
        self._num_classes = 1
        self.train = True
        self.unknown = []

    def __call__(self, item):
        """ Check to see if the Tokenizer has seen `item` before, and if so,
        return the integer class associated with it. Otherwise, if we're
        training, create a new integer class, otherwise return the 'unknown'
        class.

        """
        try:
            return self._data[item]

        except KeyError:
            if self.train:
                self._add_token(item)
                return self(item)

            else:
                # Record the unknown item, then return the unknown label
                self.unknown += [item]
                return self._data['unk']

    @property
    def num_classes(self) -> int:
        """The maximum number of assigned classes"""
        return self._num_classes

    def _add_token(self, item):
        self._num_classes += 1
        self._data[item] = self._num_classes
