import random

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import robust_pca


def add_noise(data, save=True):
    data = cv2.imread(data)

    noise_gaussian = np.random.normal(0, random.uniform(0, 1) * 255, data.shape).astype(np.uint8)
    noised_pic = cv2.add(data, noise_gaussian)

    salt_pos = [np.random.randint(0, i - 1, int(random.uniform(0, 1) * data.size)) for i in data.shape]
    noised_pic[salt_pos] = 255

    pepper_pos = [np.random.randint(0, i - 1, int(random.uniform(0, 1) * data.size)) for i in data.shape]
    noised_pic[pepper_pos] = 0

    if save:
        cv2.imwrite('noised_Lena.jpg', noised_pic)

    return noised_pic


if __name__ == "__main__":
    add_noise("Lena.bmp")
    X = torch.from_numpy(
        np.array(cv2.imread("noised_Lena.jpg", cv2.IMREAD_GRAYSCALE), dtype=np.float64),
    ).to("cuda:0")

    plt.figure(1)
    plt.imshow(X.cpu(), cmap="gray")
    plt.show()
    L, S = robust_pca.robust_pca(X, devices="cuda:0", max_iter_pass=200)
    plt.imshow(L.cpu(), cmap="gray")
    plt.savefig("noised_Lena-L.png")
    plt.show()
    plt.imshow(S.cpu(), cmap="gray")
    plt.savefig("noised_Lena-S.png")
    plt.show()



