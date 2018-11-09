import numpy as np


class OneShotBatchSampler:
    '''

    '''

    def __init__(self, labels, phase, batch_size, iteration=500):
        '''

        '''
        super(OneShotBatchSampler, self).__init__()
        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        self.n = 0
        # randomly select 2*batch_size for an iteration
        # self.query_label, self.index_list = self._gen_query_label(self.labels, self.phase)
        # batch = np.random.choice(self.index_list, size=2 * self.batch_size)
        # #Here
        # yield batch
        # return iter(batch)
        return self

    def __next__(self):
        # randomly select 2*batch_size for an iteration
        if self.n > self.iteration:
            raise StopIteration

        self.query_label, self.index_list = self._gen_query_label(self.labels, self.phase)
        batch = np.random.choice(self.index_list, size=2 * self.batch_size)
        self.n += 1
        return batch

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iteration

    def _gen_query_label(self, labels, phase):
        """
        :param labels:
        :param phase:
        :return: random query label, index list of slices with generated class available
        """
        lab_list = []
        if phase == 'train':
            lab_list = [0, 1, 2, 3, 4, 5, 6]
        elif phase == 'val':
            lab_list = [7, 8, 9]

        index_list = {i: np.unique((labels == i).nonzero()[0]) for i in lab_list}

        # find probability of selecting a query class based on how many slices it contains
        p = [1 - (len(val) / len(labels)) for val in index_list.values()]
        p = p/np.sum(p)
        # randomly generate a query label
        query_label = np.random.choice(lab_list, 1, p=p)[0]
        return query_label, index_list[query_label]
