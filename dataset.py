# -*- coding: UTF-8 -*-
import numpy as np
import data_preprocessor as io
import collections
import random

Path = collections.namedtuple('Path',['data','disp'])

class Dataset(object):

    def __init__(self,Path,EPIWidth,precision):
        self._Path = Path
        self._num_Path = Path.data.__len__()
        self._EPIWidth = EPIWidth
        self._precision = precision
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0


    def next_dataset(self):
        index = random.randint(0,self._num_Path)
        self._labels = io.read_disp(self._Path.disp[index])
        self._images = io.read_data(self._Path.data[index],self._EPIWidth)
        self._num_examples = self._images.shape[0]
        self._images, self._labels = shuffle(self._images, self._labels)


    def next_batch(self,batch_size=1):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            self.next_dataset()
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch

        return self._images[start:end], self._labels[start:end]


    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed


def get_datasets(dir,EPIWidth,disp_precision):
    Path.data, Path.disp = io.get_path_list(dir)
    train_data = Dataset(Path,EPIWidth,disp_precision)
    train_data.next_dataset()

    return train_data


def shuffle(data,disp):
    assert data.shape[0]==disp.shape[0]
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)

    return data[perm],disp[perm]