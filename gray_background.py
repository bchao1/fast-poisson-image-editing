import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils
from poisson_image_editor import PoissonImageEditor

class PoissonBackgroundChanger(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale):
        super(PoissonBackgroundChanger, self).__init__(dataset_root, solver, scale)
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)
        
    def compute_gradients(self, src, target):
        return utils.compute_laplacian(src)

    def preprocess_mask(self):
        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY)
        inner_mask, boundary_mask = utils.process_mask(self.mask)
        return self.mask, inner_mask, boundary_mask
    
    def preprocess_inputs(self):
        src_gray = utils.rgb2gray(self.src_rgb)
        return self.src_rgb, np.dstack([src_gray] * 3)

if __name__ == "__main__":
    import time

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Image scaling.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    parser.add_argument("--change_hue", default=0, type=float, help="Added hue value.")
    args = parser.parse_args()

    changer = PoissonBackgroundChanger(args.data_dir, args.solver, args.scale)

    img = changer.poisson_edit_rgb()
    
    plt.imshow(img)
    plt.show()
    
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.data_dir, f"result_{args.mode}_{args.change_hue}.png"))

    

