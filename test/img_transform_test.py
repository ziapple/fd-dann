from PIL import Image
import os
from torchvision import transforms
import numpy as np
import torch.utils.data

imgs = Image.open("../dataset/mnist_m/mnist_m_train/00000000.png").convert('RGB')
img_transform_target = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
imgs = img_transform_target(imgs)
print(imgs.shape)

dataset_target = np.random.random([7, 4])
a = np.reshape(dataset_target, [7, 1, 4])
print(a.shape)
dataloader_target = torch.utils.data.DataLoader(
    dataset=a,
    batch_size=1,
    shuffle=True,
    num_workers=4)
print(a.shape)

a = [1, 2, 3, 4, 5, 6, 7, 8]
a = np.reshape(a, (4, 2))
print(a)
