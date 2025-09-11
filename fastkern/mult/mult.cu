#include "mult.h"

__global__ void mult_kernel(const float *a, const float *b, float *c, const int m, const int n, const int p) {
    __shared__ float A[TILE_SIZE][TILE_SIZE + BANK_OFFSET];
    __shared__ float B[TILE_SIZE][TILE_SIZE + BANK_OFFSET];
    int row, col, tile_row, tile_col;
    float sum[THREAD_COARSENESS][THREAD_COARSENESS];

    // Initialize sum array
    for (int i = 0; i < THREAD_COARSENESS; ++i) {
        for (int j = 0; j < THREAD_COARSENESS; ++j) {
            sum[i][j] = 0.0f;
        }
    }

    row = THREAD_COARSENESS * (blockDim.y * blockIdx.y + threadIdx.y);
    col = THREAD_COARSENESS * (blockDim.x * blockIdx.x + threadIdx.x);

    // Loop through tiles of a and b
    for (int t = 0; t < (n + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        tile_row = t * TILE_SIZE + threadIdx.y * THREAD_COARSENESS;
        tile_col = t * TILE_SIZE + threadIdx.x * THREAD_COARSENESS;

        // Load tiles into shared memory
        for (int i = 0; i < THREAD_COARSENESS; ++i) {
            for (int j = 0; j < THREAD_COARSENESS; ++j) {
                if (row + i < m && tile_col + j < n) {
                    A[threadIdx.y * THREAD_COARSENESS + i][threadIdx.x * THREAD_COARSENESS + j] = a[(row + i) * n + (tile_col + j)];
                }
                else {
                    A[threadIdx.y * THREAD_COARSENESS + i][threadIdx.x * THREAD_COARSENESS + j] = 0.0f;
                }

                if (tile_row + i < n && col + j < p) {
                    B[threadIdx.y * THREAD_COARSENESS + i][threadIdx.x * THREAD_COARSENESS + j] = b[(tile_row + i) * p + (col + j)];
                }
                else {
                    B[threadIdx.y * THREAD_COARSENESS + i][threadIdx.x * THREAD_COARSENESS + j] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Compute partial products
        for (int i = 0; i < THREAD_COARSENESS; ++i) {
            for (int j = 0; j < THREAD_COARSENESS; ++j) {
                if (row + i < m && col + j < p) {
                    for (int k = 0; k < TILE_SIZE; ++k) {
                        sum[i][j] += A[threadIdx.y * THREAD_COARSENESS + i][k] * B[k][threadIdx.x * THREAD_COARSENESS + j];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write the results to global memory
    for (int i = 0; i < THREAD_COARSENESS; ++i) {
        for (int j = 0; j < THREAD_COARSENESS; ++j) {
            if (row + i < m && col + j < p) {
                c[(row + i) * p + (col + j)] = sum[i][j];
            }
        }
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
    
    threads = dim3(THREADS_PER_TILE, THREADS_PER_TILE);
    blocks = dim3((p + TILE_SIZE - 1) / TILE_SIZE, (m + TILE_SIZE - 1) / TILE_SIZE);
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