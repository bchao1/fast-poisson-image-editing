import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils
from poisson_image_editor import PoissonImageEditor

class PoissonColorChanger(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale):
        super(PoissonColorChanger, self).__init__(dataset_root, solver, scale)
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)
        
    def compute_gradients(self, src, target, *args, **kwargs):
        return utils.compute_laplacian(src)

    def preprocess_mask(self):
        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY)
        inner_mask, boundary_mask = utils.process_mask(self.mask)
        return self.mask, inner_mask, boundary_mask
    
    def preprocess_inputs(self, hue_change):
        src_hsv = cv2.cvtColor((self.src_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
        new_hue = src_hsv[:, :, 0] + hue_change
        src_hsv[:, :, 0] = np.where(new_hue > 180, new_hue - 180, new_hue)
        src_changed = cv2.cvtColor(src_hsv, cv2.COLOR_HSV2RGB).astype(np.float64) / 255
        return src_changed, self.src_rgb

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

    img = changer.poisson_edit_rgb(args.change_hue)
    
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.data_dir, f"result_{args.mode}_{args.change_hue}.png"))

    

