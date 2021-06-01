import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from dataset.data_loader_emd import GetLoader
from torchvision import datasets

NUM_WORKERS = 4
cuda = False


def test(dataset_name, epoch):
    assert dataset_name in ['cwru/source', 'cwru/target']

    model_root = os.path.join('..', 'models')
    emd_root = os.path.join('..', 'dataset', dataset_name)

    cudnn.benchmark = True
    batch_size = 20
    emd_height = 7
    emd_width = 400
    alpha = 0

    # 测试源数据集和目标集的准确率，都是加载目录下的test测试数据
    dataset = GetLoader(
        data_root=os.path.join(emd_root),
        data_dir="test",
        transform=None
    )

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    """ training """
    my_net = torch.load(os.path.join(
        model_root, 'emd_model_epoch_' + str(epoch) + '.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:
        # test model using target data
        data_target = data_target_iter.next()
        t_emd, t_label = data_target

        batch_size = len(t_label)

        input_emd = torch.FloatTensor(batch_size, emd_height, 1, emd_width)
        class_label = torch.LongTensor(batch_size)

        if cuda:
            t_emd = t_emd.cuda()
            t_label = t_label.cuda()
            input_emd = input_emd.cuda()
            class_label = class_label.cuda()

        input_emd.resize_as_(t_emd).copy_(t_emd)
        class_label.resize_as_(t_label).copy_(t_label)

        class_output, _ = my_net(input_data=input_emd, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(class_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, dataset_name, accu))
