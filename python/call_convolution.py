import ctypes
import numpy as np
import time
import os

# Load shared library
lib_path = os.path.abspath("../library/libmatrix.so")
lib = ctypes.CDLL(lib_path)

# Bind function
gpu_convolution = lib.gpu_convolution
gpu_convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,  # width
    ctypes.c_int,  # height
    ctypes.c_int   # kernel size K
]
gpu_convolution.restype = None

# Load CUDA runtime for synchronization
cudart = ctypes.CDLL("libcudart.so")
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# Define convolution kernels
kernels = {
    "edge_3x3": np.array([
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    ], dtype=np.int32),

    "sobel_x_3x3": np.array([
        -1,  0,  1,
        -2,  0,  2,
        -1,  0,  1
    ], dtype=np.int32),

    "gaussian_5x5": np.array([
        1,  4,  6,  4,  1,
        4, 16, 24, 16,  4,
        6, 24, 36, 24,  6,
        4, 16, 24, 16,  4,
        1,  4,  6,  4,  1
    ], dtype=np.int32),

    "edge_7x7": np.array([
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, 48, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1
    ], dtype=np.int32)
}

# Matrix sizes
image_sizes = [512, 1024, 2048, 4096, 8192]

print("CUDA Convolution Performance Results\n")

for size in image_sizes:
    width = height = size
    print(f"Image size: {width} x {height}")

    # Generate random grayscale image
    image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
    image_flat = image.ravel()

    for kernel_name, kernel in kernels.items():
        kernel = np.ascontiguousarray(kernel, dtype=np.int32)
        N = int(np.sqrt(kernel.size))

        output = np.zeros((height, width), dtype=np.int32)
        output_flat = output.ravel()

        # Warm-up run
        gpu_convolution(image_flat, kernel, output_flat, width, height, N)
        cudart.cudaDeviceSynchronize()

        # Timed run
        start = time.time()

        gpu_convolution(image_flat, kernel, output_flat, width, height, N)
        cudart.cudaDeviceSynchronize()

        end = time.time()
        elapsed_ms = (end - start) * 1000.0

        print(f"  Kernel: {kernel_name:<15} (N={N}x{N}) -> {elapsed_ms:.4f} ms")

    print("-" * 60)