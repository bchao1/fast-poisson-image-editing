import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
import pyamg

import utils

from poisson_image_editor import PoissonImageEditor

class PoissonSeamlessCloner(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale):
        super(PoissonSeamlessCloner, self).__init__(dataset_root, solver, scale)

        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)
        self.target_rgb = utils.read_image(f"{dataset_root}", "target", scale=scale, gray=False)

    def preprocess_mask(self):
        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY)
        inner_mask, boundary_mask = utils.process_mask(self.mask)
        return self.mask, inner_mask, boundary_mask

    def compute_gradients(self, src, target, gradient_mixing_mode="max", gradient_mixing_alpha=1.0):
        if gradient_mixing_mode == "max":
            Ix_src, Iy_src = utils.compute_gradient(src)
            Ix_target, Iy_target = utils.compute_gradient(target)
            I_src_amp = (Ix_src**2 + Iy_src**2)**0.5
            I_target_amp = (Ix_target**2 + Iy_target**2)**0.5
            Ix = np.where(I_src_amp > I_target_amp, Ix_src, Ix_target)
            Iy = np.where(I_src_amp > I_target_amp, Iy_src, Iy_target)
            Ixx, _ = utils.compute_gradient(Ix, forward=False)
            _, Iyy = utils.compute_gradient(Iy, forward=False)
            return Ixx + Iyy
        elif gradient_mixing_mode == "alpha":
            src_laplacian = utils.compute_laplacian(src)
            target_laplacian = utils.compute_laplacian(target)
            return gradient_mixing_alpha * src_laplacian + (1 - gradient_mixing_alpha) * target_laplacian
        else:
            raise ValueError(f"Gradient mixing mode '{mode}' not supported!")      
    
    def preprocess_inputs(self):
        return self.src_rgb, self.target_rgb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    parser.add_argument("--gradient_mixing_mode", type=str, default="max", choices=["max", "alpha"], help="Gradient mixing modes.")
    parser.add_argument("--gradient_mixing_alpha", type=float, default=1.0, help="Alpha value for gradient mixing. Mode 'max' does not depend on alpha.")
    args = parser.parse_args()

    cloner = PoissonSeamlessCloner(args.data_dir, args.solver, args.scale)

    img = cloner.poisson_edit_rgb(args.gradient_mixing_mode, args.gradient_mixing_alpha)
    
    plt.imshow(img)
    plt.show()
    #img = (img * 255).astype(np.uint8)
    #Image.fromarray(img).save(os.path.join(args.data_dir, "result.png"))

    

