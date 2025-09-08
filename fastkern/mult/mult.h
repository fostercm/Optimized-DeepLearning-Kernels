#ifndef FASTKERN_MULT_H
#define FASTKERN_MULT_H

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void mult_kernel(const float *a, const float *b, float *c, const int m, const int n);
torch::Tensor mult(const torch::Tensor &a, const torch::Tensor &b);

#endif /* FASTKERN_MULT_H */
