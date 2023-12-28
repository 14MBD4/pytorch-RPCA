import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

import robust_pca

if __name__ == "__main__":
    X = torch.from_numpy(
        np.array(cv2.imread("Lena.bmp", cv2.IMREAD_GRAYSCALE), dtype=np.float64),
    ).to("cuda:0")

    plt.figure(1)
    plt.imshow(X.cpu(), cmap="gray")
    plt.show()
    L, S = robust_pca.robust_pca(X, devices="cuda:0", max_iter_pass=200)
    plt.imshow(L.cpu(), cmap="gray")
    plt.savefig("L.png")
    plt.show()
    plt.imshow(S.cpu(), cmap="gray")
    plt.savefig("S.png")
    plt.show()
