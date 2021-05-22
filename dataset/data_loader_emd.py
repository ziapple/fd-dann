import torch.utils.data as data
from PIL import Image
import os
import numpy as np
from . import data_conf as conf


# emd数据加载器，一行一个样本(batch_size, (x_height = 7 * x_width = 400))
class GetLoader(data.Dataset):
    def __init__(self, data_root, file_name, transform=None):
        self.root = data_root
        self.transform = transform

        # 读取文件
        x = np.loadtxt(file_name)
        # 读取样本，还原
        x = x.reshape(-1, conf.EMD_IMF_SIZE, conf.X_LEN)
        # 将IMF作为深度变换（类似图片的RGB），转化成1*X_HEIGHT*X_LEN
        x_1 = np.zeros((x.shape[0], 1, conf.EMD_IMF_SIZE, conf.X_LEN))
        for j in range(x.shape[0]):
            # 每个样本做一个转置
            x_1[j, 0] = x[i].reshape(EMD_IMF_SIZE, X_LEN).T
        # 取前IMF_X_LENGTH个
        x_1 = x_1[:, :, :, :X_HEIGHT]
        n = x_1.shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, x_1[:n_split]))
        x_test = np.vstack((x_test, x_1[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (x.shape[0] - n_split)
        i += 1
        f = open(os.path.join(data_root, file_name))

        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.emd_data = []
        self.emd_labels = []

        for s_data in data_list:
            self.emd_data.append(s_data[:-3])
            self.emd_labels.append(s_data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data