import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils

class PoissonIlluminationChanger:
    def __init__(self, dataset_root, solver):
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=1, gray=True)
        self.src_rgb = utils.read_image(f"{dataset_root}", "source", scale=1, gray=False)

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

    def compute_gradients(self, src):
        Ix, Iy = utils.compute_gradient(src)
        I = np.sqrt(Ix**2 + Iy**2) # gradient norm
        alpha = 0.2 * I.mean()
        beta = 0.2
        Ix = np.power(alpha, beta) * np.power(I + 1e-8, -beta) * Ix
        Iy = np.power(alpha, beta) * np.power(I + 1e-8, -beta) * Iy
        Ixx, _ = utils.compute_gradient(Ix, forward=False)
        _, Iyy = utils.compute_gradient(Iy, forward=False)
        return Ixx + Iyy

    def poisson_illum_change_channel(self, src):
        log_src = np.log(src + 1e-8)
        log_gradients = self.compute_gradients(log_src)

        boundary_pixel_values = utils.get_masked_values(log_src, self.boundary_mask).flatten()
        inner_gradient_values = utils.get_masked_values(log_gradients, self.inner_mask).flatten()

        # Construct b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)

        # Solve Ax = b
        x = self.solver(self.A, b)
        if isinstance(x, tuple): # solvers other than spsolve
            x = x[0]
        new_log_src = np.log(np.zeros_like(src).flatten() + 1e-8)
        new_log_src[self.mask_pos] = x
        new_log_src = new_log_src.reshape(src.shape)
        new_src = np.exp(new_log_src)
        

        img = utils.get_alpha_blended_img(new_src, src, self.mask)
        img = np.clip(img, 0, 1)

        return img
    
    def poisson_illum_change_rgb(self):
        poisson_illum_changed_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_illum_changed_img_rgb.append(
                self.poisson_illum_change_channel(self.src_rgb[..., i])
            )
        return np.dstack(poisson_illum_changed_img_rgb)
    
    def poisson_illum_change_gray(self):
        src_gray = utils.rgb2gray(self.src_rgb)
        target_gray = utils.rgb2gray(self.target_rgb)
        return self.poisson_illum_change_channel(src_gray)

if __name__ == "__main__":
    import time

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--grayscale", action="store_true", help="Convert input to grayscale images.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    args = parser.parse_args()

    changer = PoissonIlluminationChanger(args.data_dir, args.solver)

    
    if args.grayscale:
        img = changer.poisson_illum_change_gray()
    else:
        img = changer.poisson_illum_change_rgb()
    
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.data_dir, "result.png"))

    

