# CUDA Matrix Multiplication & Python Acceleration Library

## Overview
This project implements and compares multiple approaches to matrix multiplication using GPU acceleration. The goal is to explore performance differences between CPU and GPU implementations of large-scale matrix operations. The project also demonstrates how to expose CUDA functionality as a shared library and integrate it with Python.

**Motivation:** 


---

## Repository Structure

```
.
├── cpu/
│   ├── matrix_cpu.c
│   └── benchmark.sh            # Optional automation script
│
├── gpu_cuda/
│   ├── notebook/
│   │   └── matrix_gpu.ipynb    # Jupyter notebook for experiments & analysis
│   │   
│   ├── matrix_gpu_naive.cu     # Naïve CUDA implementation
│   ├── matrix_gpu_tiled.cu     # Optimized CUDA implementation (tiling)
│   └── matrix_gpu_cublas.cu    # cuBLAS based matrix multiplication 
│
├── library/
│   ├── notebook/
│   └── matrix_lib.cu           # Shared library
│
├── python/
│   ├── call_matrix.py          # Python interface
│   └── call_convolution.py    
│
├── results/
│   ├── graphs/
│   └── tables/
│
└── README.md
```

---

## Features

### CPU Implementation
- Matrix multiplication in C
- Performance benchmarking across varying matrix sizes

### GPU CUDA Implementation
- Naïve CUDA matrix multiplication
- Optimized matrix multiplication using shared memory tiling
- cuBLAS-based matrix multiplication 

### Performance Analysis
- Execution time comparison between CPU and GPU
- Scaling analysis with increasing matrix/image sizes

### Python Integration
- CUDA functions exposed via shared library (`libmatrix.so`)
- Python interface using `ctypes`
- Image convolution with CUDA acceleration 

---

## Technologies Used

- C / C++
- CUDA (GPU programming)
- cuBLAS (for optimized matrix multiplication)
- Python (NumPy, ctypes)
- Google Cloud (GPU-enabled VM)

---

## How to Run

### CPU
Compile and run baseline program in the `cpu/` directory:

```
gcc matrix_cpu.c -o matrix_cpu
./matrix_cpu 512
./matrix_cpu 1024
./matrix_cpu 2048
```

### GPU (CUDA)

**Naïve CUDA**

Each GPU thread computes one output element of the result matrix.

Build and run:
```
nvcc -arch=sm_75 matrix_gpu_naive.cu -o matrix_gpu_naive
./matrix_gpu_naive 512
./matrix_gpu_naive 1024
./matrix_gpu_naive 2048
```

**Optimized CUDA (Tiling)**

Uses shared memory tiling to reduce global memory access and improve performance.

Build and run:
```
nvcc -arch=sm_75 matrix_gpu_tiled.cu -o matrix_gpu_tiled
./matrix_gpu_tiled 512
./matrix_gpu_tiled 1024
./matrix_gpu_tiled 2048
```

**cuBLAS (Production Baseline)**

Leverages NVIDIA's highly optimized BLAS library for matrix multiplication.

Build and run:
```
nvcc matrix_gpu_cublas.cu -o matrix_gpu_cublas -lcublas
./matrix_gpu_cublas 512
./matrix_gpu_cublas 1024
./matrix_gpu_cublas 2048
```

### Python + CUDA Shared Library

Run the following command inside the `library/` directory to compile shared library:

```
nvcc -Xcompiler -fPIC -shared matrix_lib.cu -o libmatrix.so
```

Run Python scripts in `python/` directory:

```
cd ../python
python call_matrix.py
python call_convolution.py
```

Example Python script:

```python
import ctypes
import numpy as np

lib = ctypes.CDLL("../library/libmatrix.so")

# Call GPU functions (example: gpu_convolution)
gpu_convolution = lib.gpu_convolution
```

---

## Key Concepts Demonstrated

- Parallel programming with CUDA
- GPU memory hierarchy (global vs. shared memory)
- Performance optimization via tiling
- Interfacing CUDA with Python
- Performance benchmarking and scalability analysis

---

## Results (Summary)

- GPU implementations significantly outperform CPU for large inputs
- Shared memory optimization provides substantial speedup over naïve CUDA
- cuBLAS achieves the highest performance due to highly optimized kernels
