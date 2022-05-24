import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils
from poisson_image_editor import PoissonImageEditor

class PoissonSeamlessTiler(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale):
        super(PoissonSeamlessTiler, self).__init__(dataset_root, solver, scale)       

    def preprocess_mask(self):
        self.mask = np.ones((self.img_h, self.img_w))
        self.boundary_mask = np.ones((self.img_h, self.img_w))
        self.boundary_mask[np.ix_(np.arange(1, self.img_h - 1), np.arange(1, self.img_w - 1))] = 0

        self.inner_mask = self.mask - self.boundary_mask
        return self.mask, self.inner_mask, self.boundary_mask

    def compute_gradients(self, src, target):
        return utils.compute_laplacian(src)
    
    def preprocess_inputs(self):
        new_boundary = np.zeros_like(self.src_rgb)
        new_boundary[0] = (self.src_rgb[0] + self.src_rgb[-1]) * 0.5
        new_boundary[-1] = (self.src_rgb[0] + self.src_rgb[-1]) * 0.5
        new_boundary[:, 0] = (self.src_rgb[:, 0] + self.src_rgb[:, -1]) * 0.5
        new_boundary[:, -1] = (self.src_rgb[:, 0] + self.src_rgb[:, -1]) * 0.5
        return self.src_rgb, new_boundary

    def tile(self, src, x_repeat, y_repeat):
        return np.tile(src, (y_repeat, x_repeat, 1))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    args = parser.parse_args()

    tiler = PoissonSeamlessTiler(args.data_dir, args.solver, args.scale)
    
    img = tiler.poisson_edit_rgb()

    orig_tile = tiler.tile(tiler.src_rgb, 2, 2)
    new_tile = tiler.tile(img, 2, 2)

    
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(orig_tile)
    axes[1].imshow(new_tile)
    plt.show()

    #orig_tile = (orig_tile * 255).astype(np.uint8)
    #Image.fromarray(orig_tile).save(os.path.join(args.data_dir, "orig_tile.png"))
    #new_tile = (new_tile * 255).astype(np.uint8)
    #Image.fromarray(new_tile).save(os.path.join(args.data_dir, "new_tile.png"))
