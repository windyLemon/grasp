import torch
from torch.utils.data import DataLoader
from utils.generator import ProposalAnchor
from model.net import GraspModel
from data.cornell import GraspData
import torch.nn.functional as F
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from utils.visual import visual
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, data, path):
        self.model = model
        self.data_loader = DataLoader(data, batch_size=1, shuffle=True, num_workers=1)
        self.proposal = ProposalAnchor()
        self.optim = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=1e-2)
        self.path = path

    def train(self):

        self.writer = SummaryWriter('runs/log')
        if os.path.exists(self.path):
            checkpoint = self.load(self.path)
            self.model.load_state_dict(checkpoint['model'])
            epoch = checkpoint['epoch']
            i = checkpoint['i']
            print('************load model***************')
        else:
            epoch = 0
            i = 0

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print('***********cuda is avaiable**********')

        while True:
            for data in self.data_loader:
                image, positive, negative = data

                if torch.cuda.is_available():
                    image = image.cuda()

                classifier, regression, anchors = self.model(image)

                label, offset, pos_anchor_indices = self.proposal(positive, anchors)

                classifier = classifier[0]
                regression = regression[0]
                # positive = positive[0]

                label = torch.from_numpy(label).long()
                offset = torch.from_numpy(offset)

                if torch.cuda.is_available():
                    label = label.cuda()
                    offset = offset.cuda()

                regression = regression[pos_anchor_indices]

                loss = self.loss(classifier, regression, label, offset)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                # self.save(epoch, i, model, self.path, loss)

                print('epoch is {}, i is {}, and loss is {}'.format(epoch, i, loss))

                i = i + 1

            epoch = epoch + 1
        # print(image.size(), positive.size(), negative.size())
        # print(offset_pos.shape, offset_theta.shape)
        # print(label.size())

    def loss(self, classifier, regression, label, offset):
        cla_loss = F.cross_entropy(classifier, label, ignore_index=-1)

        # smooth L1 loss
        diff = (regression - offset).abs()
        # print(diff.size())
        diff[:, 2] = diff[:, 2] * 10

        condition = (diff < 1).float()
        # diff = diff[diff < 1] - 0.5
        reg_loss = condition * (diff ** 2) + (1 - condition) * (diff - 0.5)
        print(reg_loss)
        print(cla_loss)

        loss = (torch.sum(cla_loss) + torch.sum(reg_loss)) / diff.size()[0]

        return loss

    def save(self, epoch, i, model, path, loss):
        save_dic = dict()
        if i % 1 == 0:
            if torch.cuda.is_available():
                model = model.cpu()
            save_dic['model'] = model.state_dict()
            save_dic['epoch'] = epoch
            save_dic['i'] = i
            torch.save(save_dic, path)

            self.writer.add_scalar('training loss', loss, epoch * len(self.data_loader) + i)

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        return checkpoint

    def test(self, image):
        if os.path.exists(self.path):
            checkpoint = self.load(self.path)
            self.model.load_state_dict(checkpoint['model'])
            print('************load model***************')

        image = torch.unsqueeze(image, dim=0)
        classifier, regression, anchor = self.model(image)
        classifier = classifier[0]
        regression = regression[0]

        classifier = F.softmax(classifier, dim=1)

        classifier = classifier.detach().numpy()
        regression = regression.detach().numpy()

        right_indices = np.argsort(classifier[:, 1])[::-1]
        num = np.where(classifier[:, 1] > 0.9)[0].shape[0]
        print(num)
        right_indices = right_indices[:num]

        right_anchor = anchor[right_indices]
        right_regression = regression[right_indices]

        # right_regression[:, 2] = right_regression[:, 2] * 10
        right = right_anchor - right_regression

        print(right_regression)


        return right


def load(path):
    image = Image.open(path)

    tf = transforms.Compose([
        transforms.Resize((480, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    image = tf(image)

    return image


g = GraspData('CornellData/data', train=True)
image, pos, neg = g[702]
#
model = GraspModel(k=6)
data = GraspData(path='CornellData/data')
t = Trainer(model, data, path='para.tar')
# t.train()

# image = load('test2 copy.jpeg')

right = t.test(image)
image = ((image.numpy().transpose(1, 2, 0) * 0.5) + 0.5)
#
visual(image, right)
