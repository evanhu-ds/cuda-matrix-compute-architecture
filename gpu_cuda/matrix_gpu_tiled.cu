#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define TILE 16

__global__ void matmul_tiled(float *a, float *b, float *c, int n) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float sum = 0.0f;

    for (int m = 0; m < (n + TILE - 1) / TILE; m++) {
        if (row < n && (m * TILE + tx) < n)
            tile_a[ty][tx] = a[row * n + m * TILE + tx];
        else
            tile_a[ty][tx] = 0.0f;

        if (col < n && (m * TILE + ty) < n)
            tile_b[ty][tx] = b[(m * TILE + ty) * n + col];
        else
            tile_b[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; k++)
            sum += tile_a[ty][k] * tile_b[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        c[row * n + col] = sum;
}

int main(int argc, char **argv) {
    int n = 512;
    if (argc > 1) {
        n = atoi(argv[1]);
    }

    printf("Tiled CUDA kernel time (N=%d): ", n);

    size_t size = n * n * sizeof(float);

    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 100 / 100.0f;
        b[i] = rand() % 100 / 100.0f;
    }

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_tiled<<<blocks, threads>>>(d_a, d_b, d_c, n);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%.3f ms\n", milliseconds);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop); 

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(c);

    return 0;
}