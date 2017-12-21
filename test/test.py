import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
from model.stylenet import Stylenet
import torchvision
import torchvision.transforms as tvt

from PIL import Image

import os


DATADIR = "test/data"

def load_image(fp):
    img = Image.open(fp)
    img = img.convert('RGB')
    return img

class TestDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        manifest = [os.path.join(root, x) for x in os.listdir(root)]
        images = [(load_image(x), 0) for x in manifest] # for now using dummy target
        self.data = images
        self.transform = transform
        self.ttransform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target).
        """
        img, target = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.ttransform is not None:
            target = self.ttransform(target)

        return img, target

    def __len__(self):
        return len(self.data)

T = tvt.Compose([
        tvt.Resize((256, 384)),
        tvt.ToTensor(),
    ])
TT = lambda x: torch.LongTensor([x])

ds = TestDataset(DATADIR, T, TT)
dl = data.DataLoader(ds, batch_size=2, shuffle=False)
for img, tgt in dl:
    print(img.size(), tgt)
