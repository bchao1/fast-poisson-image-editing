import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 

import utils
from poisson_image_editor import PoissonImageEditor

class PoissonTextureFlattener(PoissonImageEditor):
    def __init__(self, dataset_root, solver, scale, use_edge, canny_threshold, edge_dilation_kernel):
        super(PoissonTextureFlattener, self).__init__(dataset_root, solver, scale)
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)

        if use_edge:
            self.edge = utils.read_image(f"{dataset_root}", "edge", scale=scale,  gray=True)
        else:
            self.edge = cv2.Canny((utils.rgb2gray(self.src_rgb) * 255).astype(np.uint8), *canny_threshold)
            self.edge = utils.dilate_img(self.edge, edge_dilation_kernel)
        _, self.edge = cv2.threshold(self.edge, 0.5, 1, cv2.THRESH_BINARY)

    def preprocess_mask(self):
        _, self.mask = cv2.threshold(self.mask, 0.5, 1, cv2.THRESH_BINARY)
        inner_mask, boundary_mask = utils.process_mask(self.mask)
        return self.mask, inner_mask, boundary_mask

    def compute_gradients(self, src, target):
        Ix, Iy = utils.compute_gradient(src)
        Ix = self.edge * Ix
        Iy = self.edge * Iy
        Ixx, _ = utils.compute_gradient(Ix, forward=False)
        _, Iyy = utils.compute_gradient(Iy, forward=False)
        return Ixx + Iyy
        
    def preprocess_inputs(self):
        return self.src_rgb, self.src_rgb

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--use_edge", action="store_true", help="Use provided edge map. If not specified, computes depth map from source image.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    parser.add_argument("--canny_threshold", type=float, default=[100, 200], nargs="+")
    parser.add_argument("--edge_dilation_kernel", type=int, default=3)
    args = parser.parse_args()

    flattener = PoissonTextureFlattener(args.data_dir, args.solver, args.scale, args.use_edge, args.canny_threshold, args.edge_dilation_kernel)

    img = flattener.poisson_edit_rgb()

    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(os.path.join(args.data_dir, "result.png"))
    edge = (flattener.edge * 255).astype(np.uint8)
    Image.fromarray(edge).save(os.path.join(args.data_dir, "edge_canny.png"))

    

