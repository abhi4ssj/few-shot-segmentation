import numpy as np


def get_lab_list(phase):
    lab_list = []
    if phase == 'train':
        # lab_list = [0, 1, 2, 3, 4, 5, 6]
        lab_list = [1, 4]
    elif phase == 'val':
        lab_list = [5]
    return lab_list


def get_index_dict(labels, lab_list):
    index_list = {i: np.unique((labels == i).nonzero()[0]) for i in lab_list}
    p = [1 - (len(val) / len(labels)) for val in index_list.values()]
    p = p / np.sum(p)
    return index_list, p


class OneShotBatchSampler:
    '''

    '''
    
    def _gen_query_label(self):
        """
        Returns a query label uniformly from the label list of current phase. Also returns indexes of the slices which contain that label

        :return: random query label, index list of slices with generated class available
        """
        query_label = np.random.choice(self.lab_list, 1, p=self.p)[0]
        return query_label

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
        self.lab_list = get_lab_list(phase)
        self.index_dict, self.p = get_index_dict(labels, self.lab_list)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        self.n = 0
        return self

    def __next__(self):
        """
        Called on each iteration to return slices a random class label. On each iteration gets a random class label from label list and selects 2 x batch_size slices uniformly from index list
        :return: randomly select 2 x batch_size slices of a class label for the given iteration
        """
        if self.n > self.iteration:
            raise StopIteration

        self.query_label = self._gen_query_label()
        print("inside sampler")
        print(self.query_label)
        self.index_list = self.index_dict[self.query_label]
        batch = np.random.choice(self.index_list, size=2 * self.batch_size)
        self.n += 1
        return batch

    def __len__(self):
        """
        returns the number of iterations (episodes) per epoch
        :return: number os iterations
        """

        return self.iteration
