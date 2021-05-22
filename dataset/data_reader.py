from scipy.io import loadmat
import numpy as np
import random
import os
from PyEMD import EEMD
from . import data_conf as conf


# mat文件为采样样本，采样频率为12k，10秒采集一次，生成一个文件，共samples=122571采集点
# 把这samples大样本数据变成每行是EMD_IMF_SIZE*X_LEN的数据，每行可以看成是一张二维图片（高7，宽400），存储成一维数据
# 把samples样本切分成122571/X_LEN份，每份emd_sample样本长度为X_LEN，emd_sample经过emd转化为EMD_IMF_SIZE*X_LEN二维的一个数组
# 每个emd_sample二维样本数据，可以看成是每张图片样本数据


def save_data_to_emd():
    """
    读取data下面所有文件,每个mat文件对应一个emd分量文件
    :param
    :return:
    """
    eemd = EEMD()
    emd = eemd.EMD
    emd.extrema_detection = "parabol"
    for item in os.listdir(conf.DATA_DIR):
        print("start...{%s}" % item)
        short_name, _ = os.path.splitext(item)
        time_series = read_mat_data(os.path.join(conf.DATA_DIR + item))
        # 将故障样本拆分为训练样本长度
        input_length = time_series.shape[0]//X_LEN
        # 三维矩阵，记录每个输入样本长度，每个分量，信号信息,
        x = np.zeros((input_length, conf.EMD_IMF_SIZE, conf.X_LEN))
        # 获取序列最大长度，去掉信号后面的余数
        idx_last = -(time_series.shape[0] % conf.X_LEN)
        # 切片分割(input_length, x_len)
        clips = time_series[:idx_last].reshape(-1, conf.X_LEN)
        # 对每个样本切片做eemd处理
        for i in range(input_length):
            # 经过emd分解后的imf_size可能为6,7,8，统一取>=7的EMD_IMF_SIZE
            imf = eemd.eemd(clips[i])
            if imf.shape[0] >= conf.EMD_IMF_SIZE:
                x[i] = imf[:conf.EMD_IMF_SIZE]
        # x变成input_length个样本，(input_length, emd_imf_size = 7 * x_len = 400)
        a = x.reshape(input_length, -1)
        np.savetxt(os.path.join(conf.DATA_DIR + str(conf.X_LEN) + "/", short_name + '.emd'), a)
        print("save success {%s}", os.path.join(conf.DATA_DIR + str(conf.X_LEN) + "/", short_name + '.emd'))


def read_mat_data(f_path):
    """
       读取DE_time驱动端的振动数据
       DE - drive end accelerometer data
       FE - fan end accelerometer data
       BA - base accelerometer data
       time - time series data
       RPM- rpm during testing
    """
    mat_dict = loadmat(f_path)
    # 过滤DE_time这一列
    x = filter(lambda x: 'DE_time' in x, mat_dict.keys())
    # 构造数组
    x_list = [item for item in x]
    # 获取第一列
    key = x_list[0]
    # 获取n*1二维矩阵的第1列, time_serries.shape=(122571,)
    time_series = mat_dict[key][:, 0]
    return time_series


# 给定数组重新排列
def _shuffle(x, y):
    # shuffle training samples
    index = list(range(x.shape[0]))
    random.Random(0).shuffle(index)
    x = x[index]
    y = tuple(y[i] for i in index)
    return x, y


def read_emd():
    """
    读取转化后的emd_data文件,类似图片处理，每个样本按照(1, x_height, x_len)维度返回
    :return:
    """
    x_train = np.zeros((0, 1, conf.X_HEIGHT, conf.X_LEN))
    x_test = np.zeros((0, 1, conf.X_HEIGHT, conf.X_LEN))
    y_train = []
    y_test = []
    i = 0
    for item in os.listdir(os.path.join(conf.DATA_DIR, str(conf.X_LEN))):
        print("read %s" % item)
        x = np.loadtxt(conf.DATA_DIR + str(conf.X_LEN) + "/" + item)
        # 读取样本，还原
        x = x.reshape(-1, conf.X_HEIGHT, conf.X_LEN)
        # 将IMF作为深度变换（类似图片的RGB），转化成1*X_HEIGHT*X_LEN
        x_1 = np.zeros((x.shape[0], 1, conf.EMD_IMF_SIZE, conf.X_LEN))
        for j in range(x.shape[0]):
            # 每个样本做一个转置
            x_1[j, 0] = x[i].reshape(conf.EMD_IMF_SIZE, conf.X_LEN).T
        # 取前X_LENGTH个
        x_1 = x_1[:, :, :, :conf.X_HEIGHT]
        n = x_1 .shape[0]
        # 按3/4比例为训练数据，1/4为测试数据
        n_split = int((3 * n / 4))
        # 二维数组填充,增量式填充
        x_train = np.vstack((x_train, x_1[:n_split]))
        x_test = np.vstack((x_test, x_1[n_split:]))
        # [0]+[1] = [0, 1],不断累积标签
        y_train += [i] * n_split
        y_test += [i] * (x.shape[0] - n_split)
        i += 1
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    save_data_to_emd()
