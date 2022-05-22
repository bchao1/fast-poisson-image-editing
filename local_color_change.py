import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils

class PoissonColorChanger:
    def __init__(self, dataset_root, solver, scale):
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)
        self.src_rgb = utils.read_image(f"{dataset_root}", "source", scale=scale, gray=False)

        self.solver = getattr(scipy.sparse.linalg, solver)

        self.img_h, self.img_w = self.mask.shape

        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY) # fix here
        self.inner_mask, self.boundary_mask = utils.process_mask(self.mask)
        
        self.pixel_ids = utils.get_pixel_ids(self.mask) 
        self.inner_ids = utils.get_masked_values(self.pixel_ids, self.inner_mask).flatten()
        self.boundary_ids = utils.get_masked_values(self.pixel_ids, self.boundary_mask).flatten()
        self.mask_ids = utils.get_masked_values(self.pixel_ids, self.mask).flatten() # boundary + inner
        
        self.inner_pos = np.searchsorted(self.mask_ids, self.inner_ids) 
        self.boundary_pos = np.searchsorted(self.mask_ids, self.boundary_ids)
        self.mask_pos = np.searchsorted(self.pixel_ids.flatten(), self.mask_ids)


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

    def compute_gradients(self, src):
        return utils.compute_laplacian(src)

    def poisson_color_change_channel(self, src, target):
        gradients = self.compute_gradients(src)

        boundary_pixel_values = utils.get_masked_values(target, self.boundary_mask).flatten()
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
        

        img = utils.get_alpha_blended_img(new_src, target, self.mask)
        img = np.clip(img, 0, 1)

        return img
    
    def poisson_background_gray(self):
        self.A = self.construct_A_matrix()
        src_gray = utils.rgb2gray(self.src_rgb)
        poisson_color_changed_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_color_changed_img_rgb.append(
                self.poisson_color_change_channel(self.src_rgb[..., i], src_gray)
            )
        return np.dstack(poisson_color_changed_img_rgb)
    
    def poisson_color_change(self, val):
        self.A = self.construct_A_matrix()
        src_hsv = cv2.cvtColor((self.src_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        new_hue = src_hsv[:, :, 0] + val
        src_hsv[:, :, 0] = np.where(new_hue > 180, new_hue - 180, new_hue)
        src_changed = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2RGB).astype(np.float64) / 255

        poisson_color_changed_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_color_changed_img_rgb.append(
                self.poisson_color_change_channel(src_changed[..., i], self.src_rgb[..., i])
            )
        return np.dstack(poisson_color_changed_img_rgb)


if __name__ == "__main__":
    import time

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scaling.")
    parser.add_argument("--mode", type=str, help="Color change mode.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    parser.add_argument("--change_hue", default=0, type=float, help="Added hue value.")
    args = parser.parse_args()

    changer = PoissonColorChanger(args.data_dir, args.solver, args.scale)

    
    if args.mode == "gray_background":
        img = changer.poisson_background_gray()
    elif args.mode == "color_change":
        img = changer.poisson_color_change(args.change_hue)
    
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.data_dir, f"result_{args.mode}_{args.change_hue}.png"))

    

