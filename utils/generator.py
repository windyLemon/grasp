import numpy as np
import torch


class ProposalAnchor:
    def __init__(self, n_sample=128, ratio=0.25):
        self.n_sample = n_sample
        self.ratio = ratio

    def __call__(self, positive, anchor):
        n_positive = int(self.n_sample * self.ratio)
        n_negative = int(self.n_sample - n_positive)

        n = anchor.shape[0]
        self.label = np.zeros(n)
        self.position = np.zeros((n, 3))
        self.label.fill(-1)
        self.position.fill(-1)

        # positive is a tensor, anchor is a numpy array
        positive = positive.numpy()[0]
        # negative = negative.numpy()[0]
        distance_pos, distance_theta = self.cal_distance(anchor, positive)

        pos_anchor_indices, positive_indices = self.generate_pos_sample(distance_pos, distance_theta)

        print(pos_anchor_indices)
        neg_anchor_indices = np.where(np.min(np.bitwise_or(distance_pos > 60.
                                                           , distance_theta > np.pi / 3), axis=1))[0]

        if len(positive_indices) > 0:
            indices = np.random.choice(len(positive_indices), n_positive)
            pos_anchor_indices = pos_anchor_indices[indices]
            positive_indices = positive_indices[indices]

        if len(neg_anchor_indices) > 0:
            indices = np.random.choice(len(neg_anchor_indices), n_negative)
            neg_anchor_indices = neg_anchor_indices[indices]
            # negative_indices = negative_indices[indices]

        self.label[neg_anchor_indices] = 0
        self.label[pos_anchor_indices] = 1

        offset = anchor[pos_anchor_indices, :] - positive[positive_indices, :]

        return self.label, offset, pos_anchor_indices

    def cal_distance(self, anchors, positive):

        '''
        calculate distance between anchors and positive
        :param anchors: (R, 3)
        :param positive: (S, 3)
        :return: distance is (R, S), theta offset is (R, S)
        '''

        anchors = anchors[:, np.newaxis, :]

        distance = anchors - positive
        distance_pos = np.sum(distance[:, :, 0:2] ** 2, axis=2)
        distance_pos = np.sqrt(distance_pos)
        distance_theta = np.abs(distance[:, :, 2])

        return distance_pos, distance_theta

    def generate_pos_sample(self, distance_pos, distance_theta):

        pos_gt_indices = np.argmin(distance_pos, axis=1)
        pos_values = distance_pos[np.arange(distance_pos.shape[0]), pos_gt_indices]
        theta_values = distance_theta[np.arange(distance_pos.shape[0]), pos_gt_indices]

        pos_anchor_indices = np.where(np.bitwise_and(pos_values < 35., theta_values < np.pi / 6))[0]

        positive_indices = pos_gt_indices[pos_anchor_indices]

        return pos_anchor_indices, positive_indices


def generate_sample(size, k, scale):
    H, W = size

    center_x, center_y = 1 / 2, 1 / 2
    center_anchor = []
    for i in range(k):
        anchor = np.array([[center_x * scale, center_y * scale, i * np.pi / (k - 1) - np.pi / 2.]])
        center_anchor.append(anchor)
    center_anchor = np.concatenate(center_anchor, axis=0)

    anchors = []
    for i in range(H):
        for j in range(W):
            anchor = center_anchor + np.array([[i, j, 0]] * k) * scale
            anchors.append(anchor)
        # print(anchor)

    anchors = np.concatenate(anchors, axis=0)
    return anchors
