import torch.nn as nn
import torch
from models.functions import ReverseLayerF
from dataset import data_conf as conf
from torchsummary import summary
from dataset.data_loader_emd import GetLoader
import os

# emd的imf分量数，等同于图像的高度
EMD_Channels = 7
# emd分量的长度
EMD_HEIGHT = 1
EMD_WIDTH = 400

source_dataset_name = 'cwru/source'
source_emd_root = os.path.join('..', 'dataset', source_dataset_name)
# 加载目标源数据集
dataset_source = GetLoader(
    data_root=source_emd_root,
    data_dir="train",
    transform=None
)
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=20,
    drop_last=True,
    shuffle=True,
    num_workers=4)


# X输入=[batch_size = 128, channels = 1, height = 7, width = 400]
class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        # 特征提取网络
        self.feature = nn.Sequential()
        # input=[batch_size, 1, EMD_HEIGHT, EMD_WIDTH], nn.Conv2d[channels, output, height, width]
        # channels=in_channels, output表示卷积核数量，kernel_size表示卷积核大小
        self.feature.add_module('f_conv1', nn.Conv2d(EMD_Channels, 16, kernel_size=(1, 40)))
        # 归一化，BatchNorm2d(num_features)，等于上层特征数
        self.feature.add_module('f_bn1', nn.BatchNorm2d(16))
        self.feature.add_module('f_pool1', nn.MaxPool2d((1, 2)))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(16, 32, kernel_size=(1, 40)))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(32))
        self.feature.add_module('f_pool2', nn.MaxPool2d((1, 2)))
        self.feature.add_module('f_relu2', nn.ReLU(True))
        self.feature.add_module('f_conv3', nn.Conv2d(32, 64, kernel_size=(1, 40)))
        self.feature.add_module('f_bn3', nn.BatchNorm2d(64))
        self.feature.add_module('f_drop3', nn.Dropout2d())
        self.feature.add_module('f_pool3', nn.MaxPool2d((1, 2)))
        self.feature.add_module('f_relu3', nn.ReLU(True))


        # 标签分类网络
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(64 * 1 * 15, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, conf.Y_LABEL_SIZE))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        # 源域和目标域分类网络, 在特征层后加入了一项MMD适配层，用来计算源域和目标域的距离，loss就是尽量减少两个域之间的距离
        # 通过这两个域loss的训练，够提取源域和目标域的特征映射到一个共同的空间，并且接近两个特征自适应
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(64 * 1 * 15, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    # 网络前向传播，提取特征，计算分类结果和域判定结果
    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], EMD_Channels, EMD_HEIGHT, EMD_WIDTH)
        feature = self.feature(input_data)
        # 计算特征图大小
        # print(feature.shape)
        feature = feature.view(-1, 64 * 1 * 15)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


# 测试模型
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNModel().to(device)
    data_source_iter = iter(dataloader_source)
    data_source = data_source_iter.next()
    s_emd, s_label = data_source
    input_emd = torch.FloatTensor(20, EMD_Channels, EMD_HEIGHT, EMD_WIDTH)
    input_emd.resize_as_(s_emd).copy_(s_emd)
    summary(model(input_emd, 0.2), (1, 28, 28))
