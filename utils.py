import os
import cv2
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from skimage.segmentation import find_boundaries
import scipy.signal
import scipy.linalg
import scipy.sparse


def read_image(folder, name, scale=1, gray=False):
    for filename in glob.glob(folder + "/*"):
        if os.path.splitext(os.path.basename(filename))[0] == name:
            break
    img = Image.open(os.path.join(filename))
    if scale != 1:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
    if gray:
        img = img.convert("L")
    img = np.array(img)
    if len(img.shape) == 3:
        img = img[..., :3]
    return img.astype(np.float64) / 255 # only first 3

def rgb2gray(img):
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def process_mask(mask):
    boundary = find_boundaries(mask, mode="inner").astype(int)
    inner = mask - boundary
    return inner, boundary

def compute_laplacian(img):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    laplacian = scipy.signal.fftconvolve(img, kernel, mode="same")
    return laplacian

def compute_gradient(img, forward=True):
    if forward:
        kx = np.array([
            [0, 0, 0],
            [0, -1, 1],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, 0, 0],
            [0, -1, 0],
            [0, 1, 0]
        ])
    else:
        kx = np.array([
            [0, 0, 0],
            [-1, 1, 0],
            [0, 0, 0]
        ])
        ky = np.array([
            [0, -1, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])
    Gx = scipy.signal.fftconvolve(img, kx, mode="same")
    Gy = scipy.signal.fftconvolve(img, ky, mode="same")
    return Gx, Gy

def get_pixel_ids(img):
    pixel_ids = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])
    return pixel_ids

def get_masked_values(values, mask):
    assert values.shape == mask.shape
    nonzero_idx = np.nonzero(mask) # get mask 1
    return values[nonzero_idx]

def get_alpha_blended_img(src, target, alpha_mask):
    return src * alpha_mask + target * (1 - alpha_mask)

def dilate_img(img, k):
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate(img, kernel, iterations = 1)

def estimate_sparse_rank(A):
    def mv(v):
        return A @ v
    L = scipy.sparse.linalg.LinearOperator(A.shape, matvec=mv, rmatvec=mv)
    rank = scipy.linalg.interpolative.estimate_rank(L, 0.1)
    return rank
