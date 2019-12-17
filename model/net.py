import torch
import torch.nn as nn
from torchvision import models
from utils.generator import generate_sample
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils.visual import plot_feature


class GraspModel(nn.Module):
    def __init__(self, k):
        super(GraspModel, self).__init__()
        self.dense = models.densenet121(pretrained=False).features

        self.anchors = generate_sample((15, 20), k, 32)
        # n = self.anchors.shape[0]

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=(3, 3), padding=(1, 1))

        self.classifier = nn.Conv2d(512, k * 2, kernel_size=(1, 1))
        self.regression = nn.Conv2d(512, k * 3, kernel_size=(1, 1))

    def forward(self, x):
        image = x


        x = self.dense(x)

        N, C, H, W = x.size()
        x = self.conv1(x)
        x = F.relu(x)

        classifier = self.classifier(x)
        regression = self.regression(x)

        # feature = classifier[0]
        # plot_feature(classifier, image)
        # plt.imshow(feature.permute(1, 2, 0).detach().numpy()[:, :, 0])
        # plt.show()

        regression = regression.permute((0, 2, 3, 1)).contiguous().view(N, -1, 3)

        classifier = classifier.permute((0, 2, 3, 1)).contiguous().view(N, -1, 2)
        # classifier = classifier.view(N, H, W, -1, 2)

        # print(classifier.size(), regression.size())


        return classifier, regression, self.anchors
