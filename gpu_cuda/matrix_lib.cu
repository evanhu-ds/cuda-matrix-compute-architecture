#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_WIDTH 16

__global__ void matrixMultiplyTiled(float *A, float *B, float *C, int N) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    int numTiles = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int m = 0; m < numTiles; ++m) {
        int aCol = m * TILE_WIDTH + tx;
        if (Row < N && aCol < N) {
            ds_A[ty][tx] = A[Row * N + aCol];
        } else {
            ds_A[ty][tx] = 0.0f;
        }

        int bRow = m * TILE_WIDTH + ty;
        if (Col < N && bRow < N) {
            ds_B[ty][tx] = B[bRow * N + Col];
        } else {
            ds_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += ds_A[ty][k] * ds_B[k][tx];
        }

        __syncthreads();
    }

    if (Row < N && Col < N) {
        C[Row * N + Col] = Pvalue;
    }
}

extern "C" void gpu_matrix_multiply(float *h_A, float *h_B, float *h_C, int N) {
    size_t size = N * N * sizeof(float);

    float *d_A;
    float *d_B;
    float *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);

    int numBlocksX = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    int numBlocksY = (N + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 grid(numBlocksX, numBlocksY);

    matrixMultiplyTiled<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}


__global__ void convolution2D_GPU(
    unsigned char *image,
    int *kernel,
    int *output,
    int width,
    int height,
    int K
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int half = K / 2;
    int sum = 0;

    for (int ky = -half; ky <= half; ky++) {
        for (int kx = -half; kx <= half; kx++) {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                int pixel = image[iy * width + ix];
                int weight = kernel[(ky + half) * K + (kx + half)];
                sum += pixel * weight;
            }
        }
    }

    output[y * width + x] = sum;
}

extern "C" void gpu_convolution(
    unsigned char *h_image,
    int *h_kernel,
    int *h_output,
    int width,
    int height,
    int K
) {
    unsigned char *d_image;
    int *d_kernel;
    int *d_output;

    size_t img_size = width * height * sizeof(unsigned char);
    size_t kernel_size = K * K * sizeof(int);
    size_t out_size = width * height * sizeof(int);

    cudaMalloc(&d_image, img_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, out_size);

    cudaMemcpy(d_image, h_image, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    convolution2D_GPU<<<blocks, threads>>>(
        d_image, d_kernel, d_output,
        width, height, K
    );
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_image);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
