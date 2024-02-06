import torch
import numpy as np
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class_names = ['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks']

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])
# mean = np.array([0, 0, 0])
# std = np.array([1, 1, 1])
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose(
    [transforms.ToTensor(), normalize]
)

dataset = datasets.cifar.CIFAR10('data', train=False, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=25, shuffle=True)


image, label = next(iter(dataloader))

image = image.permute(0,2,3,1).contiguous()
image = image * std + mean

fig, ax = plt.subplots(nrows=5, ncols=5)
fig.suptitle('cifar10 samples', fontsize=20)
fig.set_size_inches(10,10)
#plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)

for i, (img, l) in enumerate(zip(image, label)):

    cls_name = class_names[l]
    ax[i // 5, i % 5].imshow(img)
    ax[i // 5, i % 5].set_title(cls_name)

fig.tight_layout()
plt.savefig('debug/cifar10-sample.png')
plt.show()
