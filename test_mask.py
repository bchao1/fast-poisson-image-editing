import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries

def draw_circle(L, r):
    assert r < L // 2 and r < L // 2

    center_x = L // 2
    center_y = L // 2
    
    x, y = np.meshgrid(np.arange(L), np.arange(L))
    dist_to_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    mask = (dist_to_center < r).astype(int) # True, False to 0, 1
    boundary = find_boundaries(mask, mode="inner").astype(int)
    assert (boundary * mask == boundary).all() # boundary within mask region
    inner = mask - boundary
    outer = 1 - mask

    plt.imshow(inner + outer + boundary, cmap="gray", vmin=0, vmax=1)
    plt.show()

    
if __name__ == "__main__":
    draw_circle(100, 40)
