import numpy as np

lab_list_fold = {"fold1": {"train": [2, 6, 7, 8, 9], "val": [1]},
                 "fold2": {"train": [1, 6, 7, 8, 9], "val": [2]},
                 "fold3": {"train": [1, 2, 8, 9], "val": [6, 7]},
                 "fold4": {"train": [1, 2, 6, 7], "val": [8, 9]}}


def get_lab_list(phase, fold):
    return lab_list_fold[fold][phase]


#
def get_class_slices(labels, i):
    num_slices, H, W = labels.shape
    thresh = 0.005
    total_slices = labels == i
    pixel_sum = np.sum(total_slices, axis=(1, 2)).squeeze()
    pixel_sum = pixel_sum / (H * W)
    threshold_list = [idx for idx, slice in enumerate(
        pixel_sum) if slice > thresh]
    return threshold_list


def get_index_dict(labels, lab_list):
    index_list = {i: get_class_slices(labels, i) for i in lab_list}
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

    def __init__(self, labels, phase, fold, batch_size, iteration=500):
        '''

        '''
        super(OneShotBatchSampler, self).__init__()

        self.index_list = None
        self.query_label = None
        self.batch_size = batch_size
        self.iteration = iteration
        self.labels = labels
        self.phase = phase
        self.lab_list = get_lab_list(phase, fold)
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
