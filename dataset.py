# -*- coding: UTF-8 -*-
import numpy as np
import os
from PIL import Image

data_cfg = {
    'train_size':200000,
    'validate_size':60000,
}


class Dataset(object):

    def __init__(self,images,labels,precision):
        assert images.shape[0]==labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples=images.shape[0]
        print ('num of example:%d' % (self._num_examples))

        #label是原生的浮点label
        self._images=images
        self._labels=labels
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._first=True
        print "data_shape:",images.shape
        print "label_shape:",labels.shape


    def next_batch(self,batch_size=1):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            #如果大于一次循环，重新shuffle一次
            self._images, self._labels = shuffle(self._images, self._labels)
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


def EPIextractor(image,EPIWidth):
    '''
    turn big EPI to small EPI
    :param image:np.array[9,512,3] EPI
    :return:list[512,9,EPIwidth,3]
    '''
    height = image.shape[0]
    width = image.shape[1]
    paddinghead = image[:,range(EPIWidth/2,0,-1),:]#左右颠倒
    paddinghead = paddinghead[range(9-1,-1,-1),:,:] #上下颠倒
    paddingtail = image[:,range(width-2,width-2-EPIWidth/2,-1),:]
    paddingtail = paddingtail[range(9-1,-1,-1),:,:]
    #在原图最左与最右 添加翻转过的内容作为边界填充 假设取9*16*3卷积 原图就变成9*(16/2+512+16/2)*3
    image = np.column_stack((paddinghead,image,paddingtail))
    #去均值
    mean = np.mean(np.mean(image, 0), 0)
    image = image - mean

    subEPI = [image[:,i:i+EPIWidth,:] for i in range(0,width)]
    return subEPI


def read_disp(dir):
    '''
    read disp.txt
    :param dir: the path to disp.txt
    :return: np.array[512*512] image.shape[0]*image.shape[1]
            if hot == True return hot table for classify
    '''
    disp_list = []  # 512*512
    with open(dir, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            data = line.split(' ')
            data = map(float, data)
            disp_list.append(data)
    disp = np.array(disp_list)
    disp = disp.reshape(disp.shape[0] * disp.shape[1])

    return disp

def read_data(dir,EPIWidth):
    '''
    read EPI and turn it to small EPI
    :param dir:
    :return: np.array[(512*512),9,EPIwidth,3]
    '''
    files = []
    datalist = []
    filelist = os.listdir(dir)
    for f in filelist:
        if os.path.isfile(dir+'/'+f):
            filename = dir+'/'+f
            if filename.find('.png') != -1:
                files.append(filename)
    files.sort()
    for png_path in files:
        with open(png_path) as f:
            im = Image.open(f)
            subEPI = EPIextractor(np.array(im),EPIWidth)
            datalist.append(subEPI)
    datas = np.array(datalist)
    datas = datas.reshape([datas.shape[0]*datas.shape[1],datas.shape[2],datas.shape[3],datas.shape[4]])

    return datas


def get_datasets(dir,EPIWidth,disp_precision):
    disp = read_disp(dir+'/disp.txt')
    data = read_data(dir+'/epi36_44',EPIWidth)

    data, disp = shuffle(data, disp)

    train_num = data_cfg['train_size']
    validate_num = train_num + data_cfg['validate_size']

    train_data = Dataset(data[0:train_num],disp[0:train_num],disp_precision)
    validate_data = Dataset(data[train_num:validate_num],disp[train_num:validate_num],disp_precision)

    return train_data,validate_data

def shuffle(data,disp):
    assert data.shape[0]==disp.shape[0]
    perm = np.arange(data.shape[0])
    np.random.shuffle(perm)

    return data[perm],disp[perm]