# https://github.com/fungtion/DANN

import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model import CNNModel
import numpy as np
from test import test

source_dataset_name = 'MNIST'
target_dataset_name = 'mnist_m'
source_image_root = os.path.join('..', 'dataset', source_dataset_name)
target_image_root = os.path.join('..', 'dataset', target_dataset_name)
model_root = os.path.join('..', 'models')
# GPU计算，检查Python版本和GPU的兼容
cuda = False
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
image_size = 28
n_epoch = 100
# cpu线程
NUM_WORKERS = 4

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
# 为CPU设置种子用于生成随机数
torch.manual_seed(manual_seed)

# load data
img_transform_source = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))
])

img_transform_target = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

# 下载MNIST数据集
dataset_source = datasets.MNIST(
    root='../dataset',
    train=True,
    transform=img_transform_source,
    download=True
)

# num_workers跟cpu一致
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS)

train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')

dataset_target = GetLoader(
    data_root=os.path.join(target_image_root, 'mnist_m_train'),
    data_list=train_list,
    transform=img_transform_target
)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS)


if __name__ == '__main__':
    # load model
    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    # 交叉熵
    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()

    if cuda:
        my_net = my_net.cuda()
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    # 训练网络中每个参数都保证进行梯度回传
    for p in my_net.parameters():
        p.requires_grad = True

    # training

    for epoch in range(n_epoch):
        len_dataloader = min(len(dataloader_source), len(dataloader_target))
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        i = 0
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data，训练源域数据
            data_source = data_source_iter.next()
            s_img, s_label = data_source

            # 清空过往梯度，一个batch的数据，计算一次梯度，更新一次网络
            my_net.zero_grad()
            batch_size = len(s_label)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()

            if cuda:
                s_img = s_img.cuda()
                s_label = s_label.cuda()
                input_img = input_img.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()

            # 将输入向量转化成source一样的维度
            input_img.resize_as_(s_img).copy_(s_img)
            class_label.resize_as_(s_label).copy_(s_label)

            # 训练网络返回source域的分类标签和域标签（默认为0）
            # 等价与my_net.forward(data)，在__call__中调用了self.forward()
            class_output, domain_output = my_net(input_data=input_img, alpha=alpha)
            err_s_label = loss_class(class_output, class_label)
            err_s_domain = loss_domain(domain_output, domain_label)

            # training model using target data，训练目标域数据
            data_target = data_target_iter.next()
            t_img, _ = data_target

            batch_size = len(t_img)

            input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
            # 域标签设置为1
            domain_label = torch.ones(batch_size)
            domain_label = domain_label.long()

            if cuda:
                t_img = t_img.cuda()
                input_img = input_img.cuda()
                domain_label = domain_label.cuda()

            input_img.resize_as_(t_img).copy_(t_img)

            _, domain_output = my_net(input_data=input_img, alpha=alpha)
            # 目标域判别loss
            err_t_domain = loss_domain(domain_output, domain_label)
            # 总体loss=源域标签loss+源域判别loss+目标域判别loss
            err = err_t_domain + err_s_domain + err_s_label
            # 反向传播，计算当前梯度
            err.backward()
            # 根据梯度更新网络参数
            optimizer.step()

            i += 1

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f'
                   % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)

    print('done')
