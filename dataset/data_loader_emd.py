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

        # 加载目录下的所有文件
        for item in os.listdir(os.path.join(data_root, data_dir)):
            x = np.loadtxt(item)
            # 读取样本，还原
            x = x.reshape(-1, conf.X_LEN, conf.EMD_IMF_SIZE)
            # B是球体故障0，IR是内环故障1，OR是外环故障2
            y = -1
            if item.find("B") > 0:
                y = 0
            elif item.find("IR") > 0:
                y = 1
            elif item.find("OR") > 0:
                y = 2

            # 存储数据
            self.emd_data = []
            self.emd_labels = []
            self.emd_data.extend(x)
            self.emd_labels.append([y]*len(x))

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data