import torch
import test_extension
import torch.utils.benchmark as benchmark

# Initialize
repetitions = 10000
n = 50257 * 768
a = torch.randn(n, dtype=torch.float32, device='cuda')
b = torch.randn(n, dtype=torch.float32, device='cuda')

# Warm up
for _ in range(10):
    _ = a + b
    _ = test_extension.add_tensors(a, b)
torch.cuda.synchronize()

# PyTorch Addition
pytorch_timer = benchmark.Timer(
    stmt=
    '''
    c = a + b
    ''',
    globals={'a': a, 'b': b}
)

# Custom Addition
custom_timer = benchmark.Timer(
    stmt=
    '''
    c = test_extension.add_tensors(a, b)
    ''',
    globals={'a': a, 'b': b, 'test_extension': test_extension}
)

# Measure time and bandwidth
pytorch_result = pytorch_timer.timeit(repetitions)
custom_result = custom_timer.timeit(repetitions)

pytorch_mean = pytorch_result.mean
custom_mean = custom_result.mean

bytes_moved = n * 3 * 4 # 3 tensors of 4 bytes each (float32)

pytorch_bandwidth = bytes_moved / pytorch_mean
custom_bandwidth = bytes_moved / custom_mean

# Report
print(f"PyTorch average execution time: {pytorch_mean*1e3:.3f} milliseconds")
print(f"Custom average execution time: {custom_mean*1e3:.3f} milliseconds")
print(f"PyTorch bandwidth: {pytorch_bandwidth/1e9:.3f} GB/s")
print(f"Custom bandwidth: {custom_bandwidth/1e9:.3f} GB/s")
if custom_mean < pytorch_mean:
    print(f"Custom addition is {(pytorch_mean - custom_mean) / pytorch_mean * 100:.3f}% faster than PyTorch addition")
else:
    print(f"Custom addition is {(custom_mean - pytorch_mean) / pytorch_mean * 100:.3f}% slower than PyTorch addition")
