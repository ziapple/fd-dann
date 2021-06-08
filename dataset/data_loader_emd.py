import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from . import data_conf as conf


# emd数据加载器，一行一个样本(batch_size, (x_height = 7 * x_width = 400))
class GetLoader(data.Dataset):
    def __init__(self, data_root, data_dir, transform=None):
        self.root = data_root
        self.transform = transform

        self.emd_data = []
        self.emd_labels = []
        self.n_data = 0
        # 加载目录下的所有文件
        for item in os.listdir(os.path.join(data_root, data_dir)):
            x = np.loadtxt(os.path.join(data_root, data_dir, item))
            # 读取样本，还原
            x = x.reshape(-1, conf.EMD_IMF_SIZE, conf.X_LEN)
            _x = x
            # 归一化、标准化
            # for i in range(x.shape[0]):
            #    _range = standardize(normalize(x[i].T)).T
            #    _x.append(_range)

            # B是球体故障0，IR是内环故障1，OR是外环故障2
            y = -1
            if item.find("B007") > 0:
                y = 0
            elif item.find("B014") > 0:
                y = 1
            elif item.find("B021") > 0:
                y = 2
            elif item.find("IR007") > 0:
                y = 3
            elif item.find("IR014") > 0:
                y = 4
            elif item.find("IR021") > 0:
                y = 5
            elif item.find("OR007") > 0:
                y = 6
            elif item.find("OR014") > 0:
                y = 7
            elif item.find("OR021") > 0:
                y = 8

            # 存储数据
            self.emd_data.extend(_x)
            self.emd_labels.extend([y]*len(_x))
            self.n_data = self.n_data + len(_x)
        # 转变成torch的模型输入格式（input_channels=7, height = 1, width = 400）
        self.emd_data = np.reshape(self.emd_data, [-1, conf.EMD_IMF_SIZE, 1, conf.X_LEN])
        # self.emd_data = np.reshape(self.emd_data, [-1, 1, conf.EMD_IMF_SIZE, conf.X_LEN])

    # 获取单个样本
    def __getitem__(self, item):
        imf = self.emd_data[item]
        label = self.emd_labels[item]
        if self.transform is not None:
            imf = self.transform(imf)

        return imf, label

    def __len__(self):
        return self.n_data


# 归一化
def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


# 标准化
def standardize(x):
    return (x - np.mean(x))/(np.std(x))
