import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils
from poisson_image_editor import PoissonImageEditor

class PoissonIlluminationChanger(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale):
        super(PoissonIlluminationChanger, self).__init__(dataset_root, solver, scale)
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)

    def preprocess_mask(self):
        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY) # fix here
        self.inner_mask, self.boundary_mask = utils.process_mask(self.mask)
        return self.mask, self.inner_mask, self.boundary_mask

    def compute_gradients(self, src, target):
        Ix, Iy = utils.compute_gradient(src)
        I = np.sqrt(Ix**2 + Iy**2) # gradient norm
        alpha = 0.2 * I.mean()
        beta = 0.2
        Ix = np.power(alpha, beta) * np.power(I + 1e-8, -beta) * Ix
        Iy = np.power(alpha, beta) * np.power(I + 1e-8, -beta) * Iy
        Ixx, _ = utils.compute_gradient(Ix, forward=False)
        _, Iyy = utils.compute_gradient(Iy, forward=False)
        return Ixx + Iyy
    
    def preprocess_inputs(self):
        return self.src_rgb, self.src_rgb

    # override since illumination works on log scale
    def poisson_edit_channel(self, src, target):
        log_src = np.log(src + 1e-8)
        log_gradients = self.compute_gradients(log_src, None)

        boundary_pixel_values = utils.get_masked_values(log_src, self.boundary_mask).flatten()
        inner_gradient_values = utils.get_masked_values(log_gradients, self.inner_mask).flatten()

        # Construct b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)

        # Solve Ax = b
        x = self.solver_func(self.A, b)
        new_log_src = np.log(np.zeros_like(src).flatten() + 1e-8)
        new_log_src[self.mask_pos] = x
        new_log_src = new_log_src.reshape(src.shape)
        new_src = np.exp(new_log_src)
        

        img = utils.get_alpha_blended_img(new_src, src, self.mask)
        img = np.clip(img, 0, 1)

        return img
    

if __name__ == "__main__":
    import time

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    args = parser.parse_args()

    changer = PoissonIlluminationChanger(args.data_dir, args.solver, args.scale)

    img = changer.poisson_edit_rgb()
    
    plt.imshow(img)
    plt.show()
    
    #img = (img * 255).astype(np.uint8)
    #Image.fromarray(img).save(os.path.join(args.data_dir, "result.png"))

    

