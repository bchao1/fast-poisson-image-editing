# Seamless cloning
python3 seamles_cloning.py --data_dir data/test1 --gradient_mixing_mode alpha --gradient_mixing_alpha 1.0 --solver bicg
python3 seamles_cloning.py --data_dir data/test2 --solver bicg
python3 seamles_cloning.py --data_dir data/test3 --solver bicg
python3 seamles_cloning.py --data_dir data/test4 --solver bicg

# Seamless tiling
python3 seamless_tiling.py --data_dir data/texture1 --scale 0.5 --solver bicg
python3 seamless_tiling.py --data_dir data/texture2 --scale 0.5 --solver bicg
python3 seamless_tiling.py --data_dir data/texture3 --scale 0.5 --solver bicg

# Texture flattening
python3 texture_flattening.py --data_dir data/test5 \
    --solver bicg \
    --edge_dilation_kernel 3 \
    --canny_threshold 25 100

# Local illumination change
python3 local_illumination_change.py --data_dir data/illum1 --solver bicg
