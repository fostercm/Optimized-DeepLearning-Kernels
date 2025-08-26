#ifndef FASTKERN_ADDITION_ADD_H
#define FASTKERN_ADDITION_ADD_H

#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const float *a, const float *b, float *c, int n);
torch::Tensor add(const torch::Tensor &a, const torch::Tensor &b);

#endif // FASTKERN_ADDITION_ADD_H