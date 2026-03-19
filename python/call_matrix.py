import ctypes
import numpy as np
import time
import sys
import os

lib = ctypes.cdll.LoadLibrary("../library/libmatrix.so")

lib.gpu_matrix_multiply.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int
]

N = 512

for i in range(5):
    print(f"\nTesting with matrix size N = {N}")

    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)

    start = time.time()
    lib.gpu_matrix_multiply(A.ravel(), B.ravel(), C.ravel(), N)
    end = time.time()
    elapsed_time = end - start

    print(f"Python → CUDA library time (N={N}): {elapsed_time * 1000:.4f} ms")

    N = N * 2