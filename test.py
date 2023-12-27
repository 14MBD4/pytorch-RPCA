import cv2
from matplotlib import pyplot as plt

import torch
import numpy as np

import robust_pca

if __name__ == "__main__":
    X = torch.from_numpy(
        np.array(cv2.imread("out.jpg", cv2.IMREAD_GRAYSCALE), dtype=np.float64),
    ).to("cuda:0")

    plt.figure(1)
    plt.imshow(X.cpu(), cmap="gray")
    plt.show()
    L, S = robust_pca.robust_pca(X, devices="cuda:0", max_iter_pass=50)
    plt.imshow(L.cpu(), cmap="gray")
    plt.show()
    plt.imshow(S.cpu(), cmap="gray")
    plt.show()
