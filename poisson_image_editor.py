import os
import cv2
import numpy as np
import scipy.sparse.linalg
from PIL import Image
import matplotlib.pyplot as plt
from argparse import ArgumentParser 
import pyamg
from abc import ABC, ABCMeta, abstractmethod

import utils

class PoissonImageEditor(ABC):
    def __init__(self, dataset_root, solver, scale):
        self.src_rgb = utils.read_image(f"{dataset_root}", "source", scale=scale, gray=False)

        self.solver = solver
        if solver != "multigrid":
            self.solver_func = getattr(scipy.sparse.linalg, solver)
        else:
            self.solver_func = None

        self.img_h, self.img_w = self.src_rgb.shape[:2]

    def setup(self):
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

        n1_pos = np.searchsorted(self.mask_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.mask_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.mask_ids, self.inner_ids - self.img_w)
        n4_pos = np.searchsorted(self.mask_ids, self.inner_ids + self.img_w)

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

    @abstractmethod
    def compute_gradients(self, src, target, *args, **kwargs):
        pass
    
    def solve_system(self, A, b):
        # Solve Ax = b
        if self.solver != "multigrid":
            x = self.solver_func(A, b)
            if isinstance(x, tuple): # solvers other than spsolve
                x = x[0]
        else:
            # Use multigrid solver
            ml = pyamg.ruge_stuben_solver(A)
            x = ml.solve(b, tol=1e-10)
        return x

    def poisson_edit_channel(self, src, target, *args, **kwargs):
        # passing in arguments : if args used in multiple functions? not only compute gradients?
        gradients = self.compute_gradients(src, target, *args, **kwargs)

        boundary_pixel_values = utils.get_masked_values(target, self.boundary_mask).flatten()
        inner_gradient_values = utils.get_masked_values(gradients, self.inner_mask).flatten()

        # Construct b and solve Ax = b
        b = self.construct_b(inner_gradient_values, boundary_pixel_values)
        x = self.solve_system(self.A, b)
        
        new_src = np.zeros(src.size)
        new_src[self.mask_pos] = x
        new_src = new_src.reshape(src.shape)

        poisson_edited_channel = utils.get_alpha_blended_img(new_src, target, self.mask)
        poisson_edited_channel = np.clip(poisson_edited_channel, 0, 1)
        
        return poisson_edited_channel
    
    @abstractmethod
    def preprocess_inputs(self):
        pass

    def poisson_edit_rgb(self, *args, **kwargs):
        self.setup()
        self.A = self.construct_A_matrix()
        src_rgb, target_rgb = self.preprocess_inputs()
        poisson_edited_img_rgb = []
        for i in range(self.src_rgb.shape[-1]):
            poisson_edited_img_rgb.append(
                self.poisson_edit_channel(
                    src_rgb[..., i], target_rgb[..., i], *args, **kwargs)
            )
        return np.dstack(poisson_edited_img_rgb)

if __name__ == "__main__":
    print("test")


    

