# https://github.com/fungtion/DANN

import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from dataset.data_loader_emd import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model_emd_1 import CNNModel
import numpy as np
from test_emd import test

source_dataset_name = 'cwru/source'
target_dataset_name = 'cwru/target'
source_emd_root = os.path.join('..', 'dataset', source_dataset_name)
target_emd_root = os.path.join('..', 'dataset', target_dataset_name)
model_root = os.path.join('..', 'models')
# GPU计算，检查Python版本和GPU的兼容
cuda = False
cudnn.benchmark = True
lr = 1e-3
# 批处理大小
batch_size = 20
# emd分量大小7*400
emd_channels = 7
emd_width = 400
# 迭代次数
n_epoch = 100
# cpu线程
NUM_WORKERS = 4

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
# 为CPU设置种子用于生成随机数
torch.manual_seed(manual_seed)

# 加载目标源数据集
dataset_source = GetLoader(
    data_root=source_emd_root,
    data_dir="train",
    transform=None
)
dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    drop_last=True,
    shuffle=True,
    num_workers=NUM_WORKERS)


# 加载目标数据
dataset_target = GetLoader(
    data_root=target_emd_root,
    data_dir="train",
    transform=None
)
dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    drop_last=True,
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
            s_emd, s_label = data_source

            # 清空过往梯度，一个batch的数据，计算一次梯度，更新一次网络
            my_net.zero_grad()
            batch_size = len(s_label)

            s_input_emd = torch.FloatTensor(batch_size, emd_channels, 1, emd_width)
            s_class_label = torch.LongTensor(batch_size)
            s_domain_label = torch.zeros(batch_size)
            s_domain_label = s_domain_label.long()

            if cuda:
                s_emd = s_emd.cuda()
                s_label = s_label.cuda()
                s_input_emd = s_input_emd.cuda()
                s_class_label = s_class_label.cuda()
                s_domain_label = s_domain_label.cuda()

            # 将输入向量转化成source一样的维度
            s_input_emd.resize_as_(s_emd).copy_(s_emd)
            s_class_label.resize_as_(s_label).copy_(s_label)

            # 训练网络返回source域的分类标签和域标签（默认为0）
            # 等价与my_net.forward(data)，在__call__中调用了self.forward()
            s_class_output, s_domain_output = my_net(input_data=s_input_emd, alpha=alpha)
            err_s_label = loss_class(s_class_output, s_class_label)
            err_s_domain = loss_domain(s_domain_output, s_domain_label)

            # training model using target data，训练目标域数据
            data_target = data_target_iter.next()
            t_emd, t_label = data_target
            batch_size = len(t_emd)
            t_input_emd = torch.FloatTensor(batch_size, emd_channels, 1, emd_width)
            t_class_label = torch.LongTensor(batch_size)
            # 域标签设置为1
            t_domain_label = torch.ones(batch_size)
            t_domain_label = t_domain_label.long()

            if cuda:
                t_emd = t_emd.cuda()
                t_label = t_label.cuda()
                t_input_emd = t_input_emd.cuda()
                t_class_label = t_class_label.cuda()
                t_domain_label = t_domain_label.cuda()

            t_input_emd.resize_as_(t_emd).copy_(t_emd)
            t_class_label.resize_as_(t_label).copy_(t_label)
            t_class_output, t_domain_output = my_net(input_data=t_input_emd, alpha=alpha)
            err_t_label = loss_class(t_class_output, t_class_label)
            # 目标域判别loss
            err_t_domain = loss_domain(t_domain_output, t_domain_label)
            err_t_label = loss_class(t_class_output, t_class_label)
            # 总体loss=源域标签loss+源域判别loss+目标域判别loss
            err = err_t_domain + err_s_domain + err_s_label + err_t_label
            # 反向传播，计算当前梯度
            err.backward()
            # 根据梯度更新网络参数
            optimizer.step()

            i += 1

            print('epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_label: %f, err_t_domain: %f'
                   % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                     err_s_domain.cpu().data.numpy(), err_t_label.cpu().data.numpy(), err_t_domain.cpu().data.numpy()))

        torch.save(my_net, '{0}/emd_model_epoch_{1}.pth'.format(model_root, epoch))
        # 测试源数据和目标数据的准确率
        test(source_dataset_name, epoch)
        test(target_dataset_name, epoch)

    print('done')
