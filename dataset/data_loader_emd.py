import torch.utils.data as data
from PIL import Image
import os


# emd数据加载器
class GetLoader(data.Dataset):
    def __init__(self, data_root, file_name, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data

    def read_matdata(fpath):
        """
           读取DE_time驱动端的振动数据
           DE - drive end accelerometer data
           FE - fan end accelerometer data
           BA - base accelerometer data
           time - time series data
           RPM- rpm during testing
        """
        mat_dict = loadmat(fpath)
        # 过滤DE_time这一列
        fliter_i = filter(lambda x: 'DE_time' in x, mat_dict.keys())
        # 构造数组
        fliter_list = [item for item in fliter_i]
        # 获取第一列
        key = fliter_list[0]
        # 获取n*1二维矩阵的第1列, time_serries.shape=(122571,)
        time_series = mat_dict[key][:, 0]
        return time_series
