# Poisson image editing

An implementation of the [Poisson Image Editing](https://dl.acm.org/doi/10.1145/882262.882269) paper by Perez et al. 2003.

## Features
- Seamless cloning - importing gradients
- Seamless cloning - mixing gradients

## Usage
```
python3 main.py --help
```

- `data_dir`: Folder that contains the input image files. The folder should contain image files named `mask`, `target`, and `source`. The file extension of the files can be arbitrary, as long as the files are valid image files.
- `grayscale`: Whether to perform blending on the grayscale images.
- `solver`: Linear solver to use when solving the poisson blending problem. The solver should be functions in the `scipy.sparse.linalg` library.
- `gradient_mixing_mode`: Method to mix source and target image gradients. `max` implements *3. Seamless cloning - Mixing gradients* section in the paper, while `alpha` + `gradient_mixing_alpha == 1.0` implements *3. Seamless cloning - Importing gradients* section. 
- `gradient_mixing_alpha`: Alpha to blend source and target image gradients. Has an effect only when `gradient_mixing_mode == "alpha"`. 

## Todo
- Local illumination changes
- Local color changes