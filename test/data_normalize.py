import numpy as np

def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))


def standardize(x):
    return (x - np.mean(x))/(np.std(x))


data = np.random.randn(1, 2, 4)
print('randn产生的随机数:\n', data)
for i in range(data.shape[0]):
    print("原始数据", data[i])
    # 转置处理
    a = data[i].T
    print("a=", a)
    print("归一化", normalize(a))
    print("标准化", standardize(a.T))


