#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void add_tensors_kernel(const float *a, const float *b, float *c, int n);
torch::Tensor add_tensors(const torch::Tensor &a, const torch::Tensor &b);
