# -*- coding: UTF-8 -*-
import numpy as np
import os
from PIL import Image

def EPIextractor(image,EPIWidth):
    '''
    turn big EPI to small EPI patch
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


def get_path_list(root):
    list_data = []
    list_disp = []
    train_data_path = root+'/full_data/training'
    filelist = os.listdir(train_data_path)
    for f in filelist:
        if os.path.isdir(train_data_path + '/' + f):
            foldername = train_data_path + '/' + f
            if not os.path.exists(foldername+'/epi36_44'):
                print ('create '+f+'\'s EPI')
                files = FileHelper.get_files(foldername)
                creator = EPIcreator(files)
                creator.create((36,44))
            dispname = train_data_path + '/' + f + '_disp.txt'
            foldername += '/epi36_44'
            list_data.append(foldername)
            list_disp.append(dispname)

    return list_data,list_disp


'''fang'''
class FileHelper(object):
    '''
    class to help deal with folder and File
    '''
    @classmethod
    def get_files(cls, folder, suffix='.png'):
        '''
        function:to get all files in the current folder\n
        return: tuple\n
        @suffix:file's suffix
        '''
        files = []
        filelist = os.listdir(folder)
        for f in filelist:
            if os.path.isfile(folder+'/'+f):
                filename = folder+'/'+f#get filename
                if filename.find(suffix) != -1:#filter the file
                    files.append(filename)
        files.sort()#对list的内容进行排序
        return tuple(files)#返回tuple类型

class EPIcreator(object):
    '''
    class to create origin epi file
    '''
    def __init__(self, files):
        self.files = files
        self.file_num = len(self.files)

    def create(self, block, folder=None):
        '''
        function to create origin epi file\n
        @block:图像索引闭区间\n
        @folder:to place the epi files generated\n
        default is the folder of origin images in files
        '''
        if not isinstance(block, tuple):
            raise TypeError("block should be tuple-type")
        if len(block) != 2 and block[0]<block[1]:
            raise ValueError("len(block) should be 2")
        start, end = block[0], block[1]
        #get the folder if param is None
        if folder is None:
            filename = self.files[0]
            #folder = filename[:filename.rfind('/')]
            folder = os.path.split(filename)[0]#get the folder
        epi_folder = folder+'/epi'+str(start)+'_'+str(end)+'/'
        #judge the folder exist
        if not os.path.exists(epi_folder):
            os.mkdir(epi_folder)# creat a folder for EPI
        #some prefix & suffix of file
        epi_file_prefix = 'epi_'+str(start)+'_'+str(end)+'_'
        png_suffix = '.png'
        images = self.files
        #get width and height of the image
        width, height = 0, 0
        with open(images[start], 'r') as f:
            im = Image.open(f)
            width, height = im.size
        #generate origin epi
        for h in xrange(0, height):
            epi_image = Image.new('RGB', (width, end-start+1), (255, 255, 255))
            for j in range(start, end+1):
                with open(images[j], 'r') as f:
                    im = Image.open(f)
                    for w in xrange(0, width):
                        pixel = im.getpixel((w, h))#get the pixel
                        epi_image.putpixel((w, j-start), pixel)#put the pixel into epi_image
            #保存epi图像，图像名称后缀,为截取的区间和高度，其中高度使用3位数对齐
            epi_image.save(epi_folder+epi_file_prefix+'{:0>3}'.format(h)+png_suffix)
            print 'epi'+'{:0>3}'.format(h)+' generate'
        print 'done!'