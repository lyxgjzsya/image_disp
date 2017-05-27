# -*- coding: UTF-8 -*-
import numpy as np
import os
from PIL import Image
import math
import cv2


def read_disp(dir):
    """
    :return size:512*512
    """
    disp = None
    if dir.find('.txt') != -1:
        disp_list = []
        with open(dir, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                data = line.split(' ')
                data = map(float, data)
                disp_list.append(data)
        disp = np.array(disp_list)
        disp = disp.reshape(disp.shape[0] * disp.shape[1])
    elif dir.find('.npy') != -1:
        if not os.path.exists(dir):
            path = dir[:dir.rfind('/')]
            name = path[path.rfind('/')+1:]
            path = path[:path.rfind('/')]
            disp = read_disp(path+'/'+name+'_disp.txt')
            np.save(dir,disp)
        else:
            disp = np.load(dir)
    return disp

def read_data(dir,EPIWidth):
    """
    return size:[512*512,EPIh,EPIw,channel]
    """
    EPI_u_path = dir + '/Patch_U.npy'
    if not os.path.exists(EPI_u_path):
        PatchGenerator(dir,EPIWidth,'U')
    EPI_u = np.load(EPI_u_path)

    EPI_v_path = dir + '/Patch_V.npy'
    if not os.path.exists(EPI_v_path):
        PatchGenerator(dir,EPIWidth,'V')
    EPI_v = np.load(EPI_v_path)

    return EPI_u, EPI_v


def get_path_list(root,type):
    list_data = []
    list_disp = []
    if type == 'test':
        data_path = root+'/full_data/test'
    else:
        data_path = root+'/full_data/training_oversampled'
        if not os.path.exists(data_path):
            os.mkdir(data_path)
            oversample(root)
    filelist = os.listdir(data_path)
    for f in filelist:
        if os.path.isdir(data_path + '/' + f):
            list_data.append(data_path + '/' + f)
            list_disp.append(data_path + '/' + f + '/disp_gt.npy')
    list_data.sort()
    list_disp.sort()

    return list_data,list_disp


def preprocess(image,type):
    image=image.astype(np.float32)
    shape = image.shape
    result = []
    for i in xrange(shape[0]):
        tmp = image[i]
        tmp = rgb2gray(image[i])
        tmp = sub_mean(tmp)
        #add other preprocessors...
        result.append(tmp)
    result = np.array(result)
    result = result.reshape([shape[0],shape[1],shape[2],1])
    return result

"""-------------------------以下辅助函数-----------------------"""
def sub_mean(image):
    mean = np.mean(np.mean(image,0),0)
    result = image - mean
    return result


def std(image):
    tmp = image.reshape([81,3])
    std = np.std(tmp,0)
    return std


def rgb2gray(image):
    return np.dot(image[...,:3],[0.2989,0.5870,0.1140]).astype(np.uint8)


def Patchextractor(image,EPIWidth,mode):
    """
    convert one EPI to 512*Patch with padding
    """
    assert mode == 'U' or mode == 'V'
    height = image.shape[0]
    width = image.shape[1]
    if mode=='U':
        paddinghead = image[:,range(EPIWidth/2,0,-1),:]#左右颠倒
        paddinghead = paddinghead[range(height-1,-1,-1),:,:] #上下颠倒
        paddingtail = image[:,range(width-2,width-2-EPIWidth/2,-1),:]
        paddingtail = paddingtail[range(height-1,-1,-1),:,:]

        image = np.column_stack((paddinghead,image,paddingtail))
        Patch = [image[:,i:i+EPIWidth,:] for i in range(0,width)]
    elif mode=='V':
        paddinghead = image[range(EPIWidth/2,0,-1),:,:]#上下颠倒
        paddinghead = paddinghead[:,range(width-1,-1,-1),:]#左右颠倒
        paddingtail = image[range(height-2,height-2-EPIWidth/2,-1),:,:]
        paddingtail = paddingtail[:,range(width-1,-1,-1),:]

        image = np.row_stack((paddinghead,image,paddingtail))
        Patch = [image[i:i+EPIWidth,:,:] for i in range(0,height)]

    return Patch


def PatchGenerator(folder,EPIWidth,mode):
    """
    generate xxx.npy, size:[512,512,EPI_height,EPI_width,3]
    """
    dir = folder+'/EPI-u' if mode=='U' else folder+'/EPI-v'
    Patchlist = []
    if not os.path.exists(dir):
        print ('Error EPI not exist!')
        exit(1)
    files = FileHelper.get_files(dir)
    for png_path in files:
        with open(png_path) as f:
            im = Image.open(f)
            Patch = Patchextractor(np.array(im), EPIWidth, mode)
            Patchlist.append(Patch)
    Patchset = np.array(Patchlist)
    name = folder+'/Patch_U.npy' if mode=='U' else folder+'/Patch_V.npy'
    if mode=='V':
        Patchset = np.transpose(Patchset, (0, 1, 3, 2, 4))  # h*w*EPIWidth*9*3 -> h*w*9*EPIWidth*3 卷积参数适应
        Patchset = np.transpose(Patchset, (1, 0, 2, 3, 4))  # h*w*9*EPIWidth*3 -> w*h*9*EPIWidth*3 label对应
    shape = Patchset.shape
    Patchset = Patchset.reshape([shape[0]*shape[1],shape[2],shape[3],shape[4]])
    np.save(name,Patchset)
    print name+' generated!'


def oversample(root):

    print 'start over sampling'
    dir = root + '/full_data/training'
    oversampled_dir = root + '/full_data/training_oversampled'
    filelist = FileHelper.get_files(dir, '.txt')
    print 'checking original Patch_U_V'
    for path in filelist:
        path = path[:path.rfind('_')]
        if not os.path.exists(path+'/Patch_U.npy'):
            PatchGenerator(path, 9, 'U')
        if not os.path.exists(path + '/Patch_V.npy'):
            PatchGenerator(path, 9, 'V')
    print 'original Patch checked'

    print 'counting distribution'
    count = np.zeros(115)
    for path in filelist:
        disp = read_disp(path)
        for d in disp:
            label = int((d + 4) / 0.07)
            count[label] += 1
    max_count = int(count.max())
    valid_count = np.sum(count!=0)
    total_count = int(max_count*valid_count)
    perm = np.arange(total_count)
    np.random.shuffle(perm)
    per_patch = 512 * 512
    patch_num = int(math.ceil(float(total_count) / per_patch))

    print 'ready for the tough work >_<'
    camera = ['U','V']
    for c in camera:
        print 'generating oversampled %s_data...'%c
        total_data = None
        first_flag = True
        for i in xrange(count.size):
            if count[i]!=0:
                origin = []
                for path in filelist:
                    disp = read_disp(path)
                    data_path = path[:path.rfind('_')]+'/Patch_%s.npy'%c
                    data = np.load(data_path)
                    for j in xrange(262144):
                        if int((disp[j]+4)/0.07) == i:
                            origin.append(data[j])
                origin = np.array(origin)# get the original data in label i
                origin_size = origin.shape[0]
                assert origin_size==count[i]
                print 'collected label:%d data --- original size:%d'%(i,origin_size)
                oversampled_data = []
                index = 0
                for j in xrange(max_count):# duplicate the original data
                    oversampled_data.append(origin[index])
                    index = (index + 1)%origin_size
                oversampled_data = np.array(oversampled_data)
                print 'oversampled'
                if first_flag:
                    total_data = np.copy(oversampled_data)
                    first_flag = False
                else:
                    total_data = np.concatenate((total_data,oversampled_data))
        print 'all oversampled data generated! ready to shuffle them!'
        assert total_data.shape[0]==total_count
        total_data = total_data[perm]
        for i in xrange(patch_num):
            if i != patch_num-1:
                data_patch = total_data[i*per_patch:(i+1)*per_patch]
            else:
                data_patch = total_data[i*per_patch:]
            path = oversampled_dir+'/'+'{:0>3}'.format(i)
            if not os.path.exists(path):
                os.mkdir(path)
            np.save(path+'/Patch_%s.npy'%c,data_patch)
            print path+'/Patch_%s.npy'%c+'--generated'

    print 'generating oversampled label'
    first_flag = True
    total_label = None
    for i in xrange(count.size):
        if count[i]!=0:
            oversampled_label = [i]*max_count
            if first_flag:
                total_label = np.copy(oversampled_label)
                first_flag = False
            else:
                total_label = np.concatenate((total_label,oversampled_label))
    assert total_label.shape[0]==total_count
    total_label = total_label[perm]
    for i in xrange(patch_num):
        if i!=patch_num-1:
            label_patch = total_label[i*per_patch:(i+1)*per_patch]
        else:
            label_patch = total_label[i*per_patch:]
        path = oversampled_dir + '/' + '{:0>3}'.format(i)
        np.save(path+'/disp_gt.npy',label_patch)
        print path+'/disp_gt.npy'+'--generated'
    print 'done'



class FileHelper(object):
    """
    class to help deal with folder and File
    """
    @classmethod
    def get_files(cls, folder, suffix='.png'):
        """
        function:to get all files in the current folder\n
        return: tuple\n
        @suffix:file's suffix
        """
        files = []
        filelist = os.listdir(folder)
        for f in filelist:
            if os.path.isfile(folder+'/'+f):
                filename = folder+'/'+f#get filename
                if filename.find(suffix) != -1:#filter the file
                    files.append(filename)
        files.sort()#对list的内容进行排序
        return tuple(files)#返回tuple类型


if __name__=='__main__':
    import scipy.io as sio

    root = '/home/luoyaox/Work/lightfield/full_data/stratified'
    filelist = FileHelper.get_files(root,'.txt')
    count = np.zeros(115)
    Sum = 0
    for path in filelist:
        disp = read_disp(path)
        for d in disp:
            label = int((d+4)/0.07)
            count[label]+=1
            Sum+=1
    count /= Sum

    sio.savemat('/home/luoyaox/' + 'training.mat', {'raw_output': count})




print 1



