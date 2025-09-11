#ifndef FASTKERN_MULT_H
#define FASTKERN_MULT_H

#include <torch/extension.h>
#include <cuda_runtime.h>

#define THREAD_COARSENESS 2
#define THREADS_PER_TILE 32
#define TILE_SIZE (THREADS_PER_TILE * THREAD_COARSENESS)
#define BANK_OFFSET 2

__global__ void mult_kernel(const float *a, const float *b, float *c, const int m, const int n);
torch::Tensor mult(const torch::Tensor &a, const torch::Tensor &b);

#endif /* FASTKERN_MULT_H */
