#include "add.h"

__global__ void add_kernel(const float *a, const float *b, float *c, int n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

torch::Tensor add(const torch::Tensor &a, const torch::Tensor &b) {

    torch::Tensor c = torch::empty_like(a);
    int n = a.numel();
    int threads = 512;
    int blocks = (n + threads - 1) / threads;

    add_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), 
        b.data_ptr<float>(), 
        c.data_ptr<float>(), 
        n);

    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors (CUDA)");
}