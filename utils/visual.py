import matplotlib.pyplot as plt
import numpy as np


def visual(img, pos_theta):
    plt.imshow(img)
    for position in pos_theta:
        dx = 1.
        dy = 1. * np.tan(position[2])

        dx = dx / np.sqrt(dx ** 2 + dy ** 2)
        dy = dy / np.sqrt(dx ** 2 + dy ** 2)
        plt.arrow(position[1], position[0], dx * 20, -dy * 20,
                  length_includes_head=True, head_width=5, head_length=10, fc='r', ec='b')
    plt.show()

def plot_rectangle(img, coordinates):
    plt.imshow(img)
    for coordinate in coordinates:
        pass

def plot_feature(feature, image):
    feature = feature[0]
    image = image[0]
    feature = feature.detach().numpy()
    image = image.detach().numpy()

    # print(feature.shape)

    feature = feature.transpose(1, 2, 0)
    image = image.transpose(1, 2, 0)

    k = 6
    plt.figure(0)
    plt.imshow(image)
    for i in range(1, k + 1):
        plt.figure(i)
        plt.imshow(feature[:, :, i + 1])

    plt.show()
