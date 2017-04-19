# -*- coding: UTF-8 -*-
import numpy as np
import data_preprocessor as io
import collections
import random
import tensorflow as tf

class Dataset(object):

    def __init__(self,Path,EPIWidth,precision,type):
        self._Path = Path
        self._num_of_path = Path.data.__len__()
        self._EPIWidth = EPIWidth
        self._precision = precision
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._index_of_image = 0
        self._type = type


    def next_dataset(self):
        #if self._type == 'train':
        #    self._index_of_image = random.randint(0,self._num_of_path-1)
        #elif self._type == 'test':
        self._index_of_image += 1
        if self._index_of_image == self._num_of_path:
            self._index_of_image = 0

        print ('next')
        print (self._Path.data[self._index_of_image])
        print (self._Path.disp[self._index_of_image])
        self._labels = io.read_disp(self._Path.disp[self._index_of_image])
        self._images = io.read_data(self._Path.data[self._index_of_image],self._EPIWidth)

        if self._type == 'train':
            self._images, self._labels = shuffle(self._images, self._labels)
        self._index_in_epoch = 0
        self._num_examples = self._images.shape[0]


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


    def set_index_of_image(self,index):
        self._index_of_image=index


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

    @property
    def index_of_image(self):
        return self._index_of_image

    @property
    def num_of_path(self):
        return self._num_of_path


def get_datasets(dir,EPIWidth,disp_precision,type):
    Path = collections.namedtuple('PathCollection', ['data', 'disp'])
    Path.data, Path.disp = io.get_path_list(dir,type)
    train_data = Dataset(Path,EPIWidth,disp_precision,type)
    train_data.next_dataset()

    return train_data


def shuffle(data,disp):
    assert data.shape[0]==disp.shape[0]
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)

    return data[perm],disp[perm]