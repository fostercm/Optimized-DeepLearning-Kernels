#include "mult.h"

__global__ void mult_kernel(const float *a, const float *b, float *c, const int m, const int n, const int p) {
    __shared__ float A[TILE_SIZE][TILE_SIZE];
    __shared__ float B[TILE_SIZE][TILE_SIZE];
    int row, col, tile_row, tile_col;
    float sum = 0.0f;

    row = blockDim.y * blockIdx.y + threadIdx.y;
    col = blockDim.x * blockIdx.x + threadIdx.x;

    // Loop through tiles of a and b
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        tile_row = t * TILE_SIZE + threadIdx.y;
        tile_col = t * TILE_SIZE + threadIdx.x;

        // Load tiles into shared memory
        if (row < m && tile_col < n) {
            A[threadIdx.y][threadIdx.x] = a[row * n + tile_col];
        }
        else {
            A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (tile_row < n && col < p) {
            B[threadIdx.y][threadIdx.x] = b[tile_row * p + col];
        }
        else {
            B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute partial product
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A[threadIdx.y][k] * B[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < m && col < p) {
        c[row * p + col] = sum;
    }
}

torch::Tensor mult(const torch::Tensor &a, const torch::Tensor &b) {
    int64_t m, n, p;
    dim3 threads, blocks;
    torch::Tensor c;

    m = a.size(0);
    n = a.size(1);
    p = b.size(1);

    c = torch::empty({m, p}, a.options());
    threads = dim3(TILE_SIZE, TILE_SIZE);
    blocks = dim3((p + threads.x - 1) / threads.x, (m + threads.y - 1) / threads.y);
    mult_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        m, n, p);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mult", &mult, "Multiply two tensors (CUDA)");
}