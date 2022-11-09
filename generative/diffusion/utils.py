import numpy as np
from matplotlib import pyplot as plt

def showimgs(imgs):
    #plt.figure(figsize=(20, 10))
    imgs = imgs.movedim((0, 2, 3, 1), (0, 1, 2, 3))
    imgs = np.concatenate(imgs.detach().cpu().numpy().tolist(), axis=1)
    imgs = (imgs + 1)/2
    np.clip(imgs, 0, 1)
    plt.imshow(imgs)