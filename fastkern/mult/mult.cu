#include "mult.h"

__global__ void mult_kernel(const float *a, const float *b, float *c, const int m, const int n, const int p) {
    int row, col;
    float sum;
    
    col = blockDim.x * blockIdx.x + threadIdx.x;
    row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row < m && col < p) {
        sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * p + col];
        }
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
    threads = dim3(16, 16);
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