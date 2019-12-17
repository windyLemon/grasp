import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from utils.visual import visual
from utils.generator import generate_sample


class GraspData(Dataset):
    def __init__(self, path, train=True):
        self.path = path
        self.files = []
        self.neg = []
        self.pos = []
        if train:
            directory = [self.path + '/0' + str(i) for i in range(0, 10)]
        else:
            directory = [self.path + '/10']

        for d in directory:
            self.files.extend(glob.glob(d + '/*png'))
            self.pos.extend(glob.glob(d + '/*pos*'))
            self.neg.extend(glob.glob(d + '/*neg*'))

        self.files = sorted(self.files)
        self.pos = sorted(self.pos)
        self.neg = sorted(self.neg)

    def __getitem__(self, item):
        file = self.files[item]
        neg = []
        pos = []

        image = Image.open(file)
        image = self._transforms()(image)

        with open(self.pos[item]) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                coordinates = [lines[i].split(), lines[i+1].split(),
                               lines[i+2].split(), lines[i+3].split()]
                pos.append(coordinates)

        with open(self.neg[item]) as f:
            lines = f.readlines()
            for i in range(0, len(lines), 4):
                coordinates = [lines[i].split(), lines[i + 1].split(),
                               lines[i + 2].split(), lines[i + 3].split()]
                neg.append(coordinates)

        # pos = self.position_theta(pos)
        # neg = self.position_theta(neg)
        return image, np.array(pos), neg

    def __len__(self):
        return len(self.files)

    def _transforms(self):
        tf = transforms.Compose([
            # transforms.RandomHorizontalFlip(p=1),
            # transforms.RandomCrop(400),
            transforms.Resize((480, 640)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        return tf

    def position_theta(self, positive):
        pos_the = []
        for position in positive:
            x0, y0 = position[0]
            x1, y1 = position[1]
            x2, y2 = position[2]
            x3, y3 = positive[3]

            center_x, center_y = (float(x0) + float(x2)) / 2., (float(y0) + float(y2)) / 2.
            theta = (float(y1) - float(y0)) / (float(x1) - float(x0) + 0.1)
            theta = np.arctan(theta)
            theta = -theta

            pos_the.append([center_y, center_x, theta])
        return np.array(pos_the)


g = GraspData('../CornellData/data', train=True)
image, pos, neg = g[702]
image = ((image.numpy().transpose(1, 2, 0) * 0.5) + 0.5)

print(pos.shape)
anchor = generate_sample((15, 20), 6, 32)

# print(anchor.shape)
# print(pos)
# visual(image, pos)
