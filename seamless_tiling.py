import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils

class PoissonSeamlessTiler:
    def __init__(self, dataset_root, solver, scale):
        self.src_rgb = utils.read_image(f"{dataset_root}", "texture", scale=scale, gray=False)

        self.solver = getattr(scipy.sparse.linalg, solver)

        self.img_h, self.img_w = self.src_rgb.shape[:2]
        
        self.mask = np.ones((self.img_h, self.img_w))
        self.boundary_mask = np.ones((self.img_h, self.img_w))
        self.boundary_mask[np.ix_(np.arange(1, self.img_h - 1), np.arange(1, self.img_w - 1))] = 0

        self.inner_mask = self.mask - self.boundary_mask
        
        self.pixel_ids = utils.get_pixel_ids(self.mask) 
        self.inner_ids = utils.get_masked_values(self.pixel_ids, self.inner_mask).flatten()
        self.boundary_ids = utils.get_masked_values(self.pixel_ids, self.boundary_mask).flatten()
        self.mask_ids = utils.get_masked_values(self.pixel_ids, self.mask).flatten() # boundary + inner
        
        self.inner_pos = np.searchsorted(self.mask_ids, self.inner_ids) 
        self.boundary_pos = np.searchsorted(self.mask_ids, self.boundary_ids)
        self.mask_pos = np.searchsorted(self.pixel_ids.flatten(), self.mask_ids)

        self.A = self.construct_A_matrix()

    def construct_A_matrix(self):
        A = scipy.sparse.lil_matrix((len(self.mask_ids), len(self.mask_ids)))

        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w )
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

        A[self.inner_pos, n1_pos] = 1
        A[self.inner_pos, n2_pos] = 1
        A[self.inner_pos, n3_pos] = 1
        A[self.inner_pos, n4_pos] = 1
        A[self.inner_pos, self.inner_pos] = -4 

        A[self.boundary_pos, self.boundary_pos] = 1
        return A.tocsr()
    
    def construct_b(self, inner_gradient_values, boundary_pixel_values):
        b = np.zeros(len(self.mask_ids))
        b[self.inner_pos] = inner_gradient_values
        b[self.boundary_pos] = boundary_pixel_values
        return b

    def poisson_tile_channel(self, src):
        gradients = utils.compute_laplacian(src)

        new_boundary = np.zeros_like(src)
        new_boundary[0] = (src[0] + src[-1]) * 0.5
        new_boundary[-1] = (src[0] + src[-1]) * 0.5
        new_boundary[:, 0] = (src[:, 0] + src[:, -1]) * 0.5
        new_boundary[:, -1] = (src[:, 0] + src[:, -1]) * 0.5

        boundary_pixel_values = utils.get_masked_values(new_boundary, self.boundary_mask).flatten()
        inner_gradient_values = utils.get_masked_values(gradients, self.inner_mask).flatten()

        # Construct b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)

        # Solve Ax = b
        x = self.solver(self.A, b)
        if isinstance(x, tuple): # solvers other than spsolve
            x = x[0]
        new_src = np.zeros_like(src).flatten()
        new_src[self.mask_pos] = x
        new_src = new_src.reshape(src.shape)
        new_src = np.clip(new_src, 0, 1)
        return new_src
    
    def poisson_tile_rgb(self):
        new_src_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            new_src_rgb.append(self.poisson_tile_channel(self.src_rgb[..., i]))
        return np.dstack(new_src_rgb)
    
    def poisson_tile_gray(self):
        src_gray = utils.rgb2gray(self.src_rgb)
        return self.poisson_tile_channel(src_gray)
    
    def tile(self, src, x_repeat, y_repeat):
        return np.tile(src, (y_repeat, x_repeat, 1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--grayscale", action="store_true", help="Convert input to grayscale images.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    args = parser.parse_args()

    tiler = PoissonSeamlessTiler(args.data_dir, args.solver, args.scale)

    if args.grayscale:
        img = tiler.poisson_tile_gray()
    else:
        img = tiler.poisson_tile_rgb()


    orig_tile = tiler.tile(tiler.src_rgb, 2, 2)
    new_tile = tiler.tile(img, 2, 2)

    orig_tile = (orig_tile * 255).astype(np.uint8)
    Image.fromarray(orig_tile).save(os.path.join(args.data_dir, "orig_tile.png"))

    new_tile = (new_tile * 255).astype(np.uint8)
    Image.fromarray(new_tile).save(os.path.join(args.data_dir, "new_tile.png"))
