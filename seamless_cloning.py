import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
import pyamg

import utils

class PoissonSeamlessCloner:
    #@profile
    def __init__(self, dataset_root, solver, scale):
        self.mask = utils.read_image(f"{dataset_root}", "mask", scale=scale, gray=True)
        self.src_rgb = utils.read_image(f"{dataset_root}", "source", scale=scale, gray=False)
        self.target_rgb = utils.read_image(f"{dataset_root}", "target", scale=scale,  gray=False)
        
        self.solver = solver
        self.solver_func = getattr(scipy.sparse.linalg, solver)

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

    #@profile
    def construct_C_matrix(self):
        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w)
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

        #C = scipy.sparse.lil_matrix((3 * len(self.mask_ids), 3 * len(self.mask_ids)))
        #for i in range(3):
        #    offset = i * len(self.mask_ids)
        #    C[offset + self.inner_pos, offset + n1_pos] = 1
        #    C[offset + self.inner_pos, offset + n2_pos] = 1
        #    C[offset + self.inner_pos, offset + n3_pos] = 1
        #    C[offset + self.inner_pos, offset + n4_pos] = 1
        #    C[offset + self.inner_pos, offset + self.inner_pos] = -4 
        #    C[offset + self.boundary_pos, offset + self.boundary_pos] = 1
        #C = C.tocsr()

        l = len(self.mask_ids)
        row_ids = np.concatenate([
            self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.boundary_pos,
            l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.inner_pos, l + self.boundary_pos,
            2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.inner_pos, 2 * l + self.boundary_pos
        ])
        col_ids = np.concatenate([
            n1_pos, n2_pos, n3_pos, n4_pos, self.inner_pos, self.boundary_pos,
            l + n1_pos, l + n2_pos, l + n3_pos, l + n4_pos, l + self.inner_pos, l + self.boundary_pos,
            2 * l + n1_pos, 2 * l + n2_pos, 2 * l + n3_pos, 2 * l + n4_pos, 2 * l + self.inner_pos, 2 * l + self.boundary_pos
        ])
        data = ([1] * len(self.inner_pos) * 4 + [-4] * len(self.inner_pos) + [1] * len(self.boundary_pos)) * 3
        C = scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(3 * len(self.mask_ids), 3 * len(self.mask_ids)))

        return C

    def construct_A_matrix(self):

        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w)
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

        #row_ids = np.concatenate([self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.inner_pos, self.boundary_pos])
        #col_ids = np.concatenate([n1_pos, n2_pos, n3_pos, n4_pos, self.inner_pos, self.boundary_pos])
        #data = [1] * len(self.inner_pos) * 4 + [-4] * len(self.inner_pos) + [1] * len(self.boundary_pos)
 
        #A = scipy.sparse.csr_matrix((data, (row_ids, col_ids)), shape=(len(self.mask_ids), len(self.mask_ids)))

        A = scipy.sparse.lil_matrix((len(self.mask_ids), len(self.mask_ids)))
        A[self.inner_pos, n1_pos] = 1
        A[self.inner_pos, n2_pos] = 1
        A[self.inner_pos, n3_pos] = 1
        A[self.inner_pos, n4_pos] = 1
        A[self.inner_pos, self.inner_pos] = -4 
        A[self.boundary_pos, self.boundary_pos] = 1
        A = A.tocsr()
        
        return A
    
    def construct_b(self, inner_gradient_values, boundary_pixel_values):
        b = np.zeros(len(self.mask_ids))
        b[self.inner_pos] = inner_gradient_values
        b[self.boundary_pos] = boundary_pixel_values
        return b
    
    #@profile
    def compute_mixed_gradients(self, src, target, mode="max", alpha=1.0):
        if mode == "max":
            Ix_src, Iy_src = utils.compute_gradient(src)
            Ix_target, Iy_target = utils.compute_gradient(target)
            I_src_amp = (Ix_src**2 + Iy_src**2)**0.5
            I_target_amp = (Ix_target**2 + Iy_target**2)**0.5
            Ix = np.where(I_src_amp > I_target_amp, Ix_src, Ix_target)
            Iy = np.where(I_src_amp > I_target_amp, Iy_src, Iy_target)
            Ixx, _ = utils.compute_gradient(Ix, forward=False)
            _, Iyy = utils.compute_gradient(Iy, forward=False)
            return Ixx + Iyy
        elif mode == "alpha":
            src_laplacian = utils.compute_laplacian(src)
            target_laplacian = utils.compute_laplacian(target)
            return alpha * src_laplacian + (1 - alpha) * target_laplacian
        else:
            raise ValueError(f"Gradient mixing mode '{mode}' not supported!")
    
    #@profile
    def poisson_blend_channel(self, src, target, gradient_mixing_mode, gradient_mixing_alpha):
        mixed_gradients = self.compute_mixed_gradients(src, target, gradient_mixing_mode, gradient_mixing_alpha)

        boundary_pixel_values = utils.get_masked_values(target, self.boundary_mask).flatten()
        inner_gradient_values = utils.get_masked_values(mixed_gradients, self.inner_mask).flatten()

        # Construct b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)

        # Solve Ax = b
        if self.solver != "multigrid":
            x = self.solver_func(self.A, b)
            if isinstance(x, tuple): # solvers other than spsolve
                x = x[0]
        else:
            # Use multigrid solver
            ml = pyamg.ruge_stuben_solver(self.A)
            x = ml.solve(b, tol=1e-10)
        new_src = np.zeros(src.size)
        new_src[self.mask_pos] = x
        new_src = new_src.reshape(src.shape)
        poisson_blended_img = utils.get_alpha_blended_img(new_src, target, self.mask)

        poisson_blended_img = np.clip(poisson_blended_img, 0, 1)
        
        return poisson_blended_img
    
    #@profile
    def poisson_blend_rgb_v2(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.C = self.construct_C_matrix()
        b_full = []
        for i in range(self.src_rgb.shape[-1]):
            src = self.src_rgb[..., i]
            target = self.target_rgb[..., i]
            mixed_gradients = self.compute_mixed_gradients(src, target, gradient_mixing_mode, gradient_mixing_alpha)
            boundary_pixel_values = utils.get_masked_values(target, self.boundary_mask).flatten()
            inner_gradient_values = utils.get_masked_values(mixed_gradients, self.inner_mask).flatten()
            b = self.construct_b(inner_gradient_values, boundary_pixel_values)
            b_full.append(b)
        b_full = np.concatenate(b_full)
        x = self.solver(self.C, b_full)#[0]
        x = x.reshape(3, -1).T
        new_src = np.zeros((self.img_w * self.img_h, 3))
        new_src[self.mask_pos, :] = x
        new_src = new_src.reshape(self.src_rgb.shape)
        poisson_blended_img = utils.get_alpha_blended_img(new_src, self.target_rgb, np.expand_dims(self.mask, -1))

        poisson_blended_img = np.clip(poisson_blended_img, 0, 1)
        
        return poisson_blended_img

    #@profile
    def poisson_blend_rgb(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.A = self.construct_A_matrix()
        poisson_blended_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_blended_img_rgb.append(
                self.poisson_blend_channel(
                    self.src_rgb[..., i], self.target_rgb[..., i],
                    gradient_mixing_mode, gradient_mixing_alpha
                )
            )
        return np.dstack(poisson_blended_img_rgb)
    
    def poisson_blend_gray(self, gradient_mixing_mode, gradient_mixing_alpha):
        self.A = self.construct_A_matrix()
        src_gray = utils.rgb2gray(self.src_rgb)
        target_gray = utils.rgb2gray(self.target_rgb)
        return self.poisson_blend_channel(src_gray, target_gray, gradient_mixing_mode, gradient_mixing_alpha)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Folder of mask, source, and target image files.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scaling image height and width.")
    parser.add_argument("--grayscale", action="store_true", help="Convert input to grayscale images.")
    parser.add_argument("--solver", type=str, default="spsolve", help="Linear system solver.")
    parser.add_argument("--gradient_mixing_mode", type=str, default="max", choices=["max", "alpha"], help="Gradient mixing modes.")
    parser.add_argument("--gradient_mixing_alpha", type=float, default=1.0, help="Alpha value for gradient mixing. Mode 'max' does not depend on alpha.")
    args = parser.parse_args()

    cloner = PoissonSeamlessCloner(args.data_dir, args.solver, args.scale)

    if args.grayscale:
        img = cloner.poisson_blend_gray(args.gradient_mixing_mode, args.gradient_mixing_alpha)
    else:
        img = cloner.poisson_blend_rgb(args.gradient_mixing_mode, args.gradient_mixing_alpha)
        
    #img = (img * 255).astype(np.uint8)
    #Image.fromarray(img).save(os.path.join(args.data_dir, "result.png"))

    

