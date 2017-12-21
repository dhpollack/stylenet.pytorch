import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as tvt
import numpy as np
from PIL import Image

import os

def load_image(fp):
    img = Image.open(fp)
    img = img.convert('RGB')
    return img

class Fashion144kDataset(data.Dataset):

    T = tvt.Compose([
            tvt.Resize((256, 384)),
            tvt.ToTensor(),
        ])

    def __init__(self, root, trans=None, ttrans=None,
                 features_type="single"):
        photosdir = os.path.join(root, "photos")
        manifest = [os.path.join(photosdir, x) for x in os.listdir(photosdir)]
        feat_fp = "feat_sin.npy" if features_type == "single" else "feat_col.npy"
        labels = np.load(os.path.join(root, "feat", feat_fp))
        labels = torch.from_numpy(labels.astype(np.int32)).long()
        dummy_cat = torch.zeros((labels.size(0), 1)).long()
        labels = torch.cat((dummy_cat, labels), dim=1)
        data = [(fp, label) for fp, label in zip(manifest, labels)] # for now using dummy target
        self.data = data
        self.n_feats = labels.size(1)
        self.trans = trans if trans is not None else self.T
        self.ttrans = ttrans

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target).
        """
        img_fp, target = self.data[index]
        img = load_image(img_fp)
        if self.trans is not None:
            img = self.trans(img)
        if self.ttrans is not None:
            target = self.ttrans(target)

        return img, target

    def __len__(self):
        return len(self.data)
