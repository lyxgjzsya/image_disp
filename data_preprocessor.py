# -*- coding: UTF-8 -*-
import numpy as np
import os
from PIL import Image

def read_disp(dir):
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


def read_data(dir,EPIWidth,UV_Plus=False):
#    EPI_u_path = dir + '/Patch_U.npy'
#    if not os.path.exists(EPI_u_path):
#        PatchGenerator(dir,EPIWidth,'U')
#    EPI_u = np.load(EPI_u_path)
#    data = EPI_u
    if UV_Plus:
        EPI_v_path = dir + '/Patch_V.npy'
        if not os.path.exists(EPI_v_path):
            PatchGenerator(dir,EPIWidth,'V')
        EPI_v = np.load(EPI_v_path)
        #EPI_v对应的label要和u统一则需先转置
        EPI_v = np.transpose(EPI_v,(0,1,3,2,4))#h*w*EPIWidth*9*3 -> h*w*9*EPIWidth*3 针对卷积参数适应
        EPI_v = np.transpose(EPI_v,(1,0,2,3,4))#h*w*9*EPIWidth*3 -> w*h*9*EPIWidth*3 针对label对应
        #通道合并
#        shape = EPI_u.shape
#        EPI_u = EPI_u.reshape([shape[0]*shape[1]*shape[2]*shape[3],shape[4]])
#        EPI_v = EPI_v.reshape([shape[0]*shape[1]*shape[2]*shape[3],shape[4]])
#        data = np.column_stack((EPI_u,EPI_v))
#        data = data.reshape([shape[0],shape[1],shape[2],shape[3],shape[4]*2])
    data = EPI_v
    data = data.reshape([data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4]])

    return data


def get_path_list(root,type):
    list_data = []
    list_disp = []
    if type == 'train':
        train_data_path = root+'/full_data/training'
    elif type == 'test':
        train_data_path = root+'/full_data/additional'
    else:
        train_data_path = root+'???'
    filelist = os.listdir(train_data_path)
    for f in filelist:
        if os.path.isdir(train_data_path + '/' + f):
            foldername = train_data_path + '/' + f
            dispname = train_data_path + '/' + f + '_disp.txt'
            list_data.append(foldername)
            list_disp.append(dispname)

    return list_data,list_disp


def preprocess(image):
    image=image.astype(np.float32)
    count = image.shape[0]
    for i in xrange(count):
        image[i] = reduce_mean(image[i])
        pass
    return image


'''--------------以下辅助函数-----------------------'''
def reduce_mean(image):
    mean = np.mean(np.mean(image,0),0)
    image = image - mean
    return image


def EPIextractor(image,EPIWidth,mode):
    assert mode == 'U' or mode == 'V'
    height = image.shape[0]
    width = image.shape[1]
    if mode=='U':
        paddinghead = image[:,range(EPIWidth/2,0,-1),:]#左右颠倒
        paddinghead = paddinghead[range(height-1,-1,-1),:,:] #上下颠倒
        paddingtail = image[:,range(width-2,width-2-EPIWidth/2,-1),:]
        paddingtail = paddingtail[range(height-1,-1,-1),:,:]

        image = np.column_stack((paddinghead,image,paddingtail))
        subEPI = [image[:,i:i+EPIWidth,:] for i in range(0,width)]
    elif mode=='V':
        paddinghead = image[range(EPIWidth/2,0,-1),:,:]#上下颠倒
        paddinghead = paddinghead[:,range(width-1,-1,-1),:]#左右颠倒
        paddingtail = image[range(height-2,height-2-EPIWidth/2,-1),:,:]
        paddingtail = paddingtail[:,range(width-1,-1,-1),:]

        image = np.row_stack((paddinghead,image,paddingtail))
        subEPI = [image[i:i+EPIWidth,:,:] for i in range(0,height)]

    return subEPI


def PatchGenerator(folder,EPIWidth,mode):
    dir = folder+'/EPI-u' if mode=='U' else folder+'/EPI-v'
    files = []
    datalist = []
    if not os.path.exists(dir):
        print ('Error EPI not exist!')
        exit(1)
    filelist = os.listdir(dir)
    for f in filelist:
        if os.path.isfile(dir + '/' + f):
            filename = dir + '/' + f
            if filename.find('.png') != -1:
                files.append(filename)
    files.sort()
    for png_path in files:
        with open(png_path) as f:
            im = Image.open(f)
            subEPI = EPIextractor(np.array(im), EPIWidth, mode)
            datalist.append(subEPI)
    datas = np.array(datalist)
#    datas = datas.reshape([datas.shape[0] * datas.shape[1], datas.shape[2], datas.shape[3], datas.shape[4]])
    name = folder+'/Patch_U.npy' if mode=='U' else folder+'/Patch_V.npy'
    np.save(name,datas)


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