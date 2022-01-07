import nrrd
from sklearn.model_selection import train_test_split
import os
import paddle
import numpy as np

def data_split(root_path, val_ratio):
    filelists = os.listdir(root_path)
    train_filelists, val_filelists = train_test_split(filelists, test_size=val_ratio, random_state=42)
    print("Total Nums: {}, train: {}, val: {}".format(len(filelists), len(train_filelists), len(val_filelists)))
    return (train_filelists, val_filelists)

class Data_Reader(paddle.io.Dataset):
    def __init__(self, data_root, file_list, w, h, d, mode='train'):
        self.file_list = file_list
        self.mode = mode
        self.data_root = data_root
        self.w = w
        self.h = h
        self.d = d

    def data_load(self, data_path):
        smooth = 1e-5
        data, p = nrrd.read(os.path.join(self.data_root,data_path,data_path+'.nrrd'))
        max = np.max(data)
        min = np.min(data)
        data = (data-min+smooth)/(max-min+smooth)
        return  data

    def label_load(self, data_path):
        mask, p = nrrd.read(os.path.join(self.data_root,data_path,'label.'+data_path+'.nrrd'))
        return  mask

    def train_random_crop(self, data, label, w, h, d):
        z, x, y = data.shape
        xw = np.random.randint(0, x - w, 1)[0]
        yh = np.random.randint(0, y - h, 1)[0]
        zd = np.random.randint(0, z - d, 1)[0]
        data = data[zd:d + zd, xw:w + xw, yh:h + yh]
        label = label[zd:d + zd, xw:w + xw, yh:h + yh]
        data = data[np.newaxis, :]
        label = label[np.newaxis, :]
        return (paddle.to_tensor(data, dtype = 'float32'), paddle.to_tensor(label,  dtype = 'float32'))

    def inference_random_crop(self, data, w, h, d):
        z, x, y = data.shape
        xw = np.random.randint(0, x - w, 1)[0]
        yh = np.random.randint(0, y - h, 1)[0]
        zd = np.random.randint(0, z - d, 1)[0]
        data = data[zd:d + zd, xw:w + xw, yh:h + yh]
        data = data[np.newaxis, :]
        return paddle.to_tensor(data, dtype = 'float32')

    def __getitem__(self, idx):
        file = self.file_list[idx]
        image = self.data_load(str(file))
        if self.mode == 'train':
            label = self.label_load(str(file))
            data = self.train_random_crop(image, label, self.w, self.h, self.d)
        else:
            data = self.inference_random_crop(image, self.w, self.h, self.d)
        return data

    def __len__(self):
        return len(self.file_list)

