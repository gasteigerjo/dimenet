import numpy as np


class DataProvider:
    def __init__(self, data_container, ntrain, nvalid, batch_size=1,
                 seed=None, randomized=False):
        self.data_container = data_container
        self._ndata = len(data_container)
        self.nsamples = {'train': ntrain, 'val': nvalid, 'test': len(data_container) - ntrain - nvalid}
        self.batch_size = batch_size

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))
        if randomized:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        # Store indices of training, validation and test data
        self.idx = {'train': all_idx[0:ntrain],
                    'val': all_idx[ntrain:ntrain+nvalid],
                    'test': all_idx[ntrain+nvalid:]}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch(self, split):
        """Return a batch of samples from the training set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if start == 0 and split == 'train':
            self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.data_container[self.idx[split][start:end]]

    def get_batch_fn(self, split):
        return lambda: self.get_batch(split)
