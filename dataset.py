# -*- coding: UTF-8 -*-
import numpy as np
import data_preprocessor as io
import collections

class Dataset(object):

    def __init__(self,Path,EPIWidth,precision,type):
        self._Path = Path
        self._num_of_path = Path.data.__len__()
        self._EPIWidth = EPIWidth
        self._precision = precision
        self._num_examples = 0
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._index_of_image = -1
        self._type = type
        self._complete = False


    def next_dataset(self):
        self._index_of_image += 1
        if self._index_of_image == self._num_of_path:
            self._index_of_image = 0
            self._complete = True

        labels = io.read_disp(self._Path.disp[self._index_of_image])
        image_u, image_v = io.read_data(self._Path.data[self._index_of_image],self._EPIWidth)

        image_u = io.preprocess(image_u,self._type)
        image_v = io.preprocess(image_v,self._type)

        self._index_in_epoch = 0
        self._num_examples = image_u.shape[0]
        self._labels = labels
        self._u = image_u
        self._v = image_v

        if self._type == 'test':
            print ('test data set:')
            print (self.get_data_name())
        if self._type == 'train':
            print ('train data set:')
            print (self.get_data_name())


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
        return self._u[start:end], self._v[start:end], self._labels[start:end]


    def set_index_of_image(self,index):
        self._index_of_image=index


    def get_data_name(self):
        dir = self._Path.data[self._index_of_image]
        return dir[dir.rfind('/')+1:]


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

    @property
    def complete(self):
        return self._complete


def get_datasets(dir,EPIWidth,disp_precision,type):
    Path = collections.namedtuple('PathCollection', ['data', 'disp'])
    Path.data, Path.disp = io.get_path_list(dir,type)
    train_data = Dataset(Path,EPIWidth,disp_precision,type)
    train_data.next_dataset()

    return train_data


def shuffle(data_u,data_v,disp):
    assert data_u.shape[0]==disp.shape[0] and data_v.shape[0]==disp.shape[0]
    perm = np.arange(data_u.shape[0])
    np.random.shuffle(perm)

    return data_u[perm],data_v[perm],disp[perm]