import torch
import fastkern as fk
import torch.utils.benchmark as benchmark

# Benchmarking function
def benchmark_add(dim1: int, dim2: int, repetitions: int) -> None:
    a = torch.randn((dim1, dim2), dtype=torch.float32, device='cuda')
    b = torch.randn((dim1, dim2), dtype=torch.float32, device='cuda')

    # Warm up
    for _ in range(10):
        _ = a + b
        _ = fk.add(a, b)
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
        c = fk.add(a, b)
        ''',
        globals={'a': a, 'b': b, 'fk': fk}
    )

    # Measure time and bandwidth
    pytorch_result = pytorch_timer.timeit(repetitions)
    custom_result = custom_timer.timeit(repetitions)

    pytorch_mean = pytorch_result.mean
    custom_mean = custom_result.mean

    bytes_moved = dim1 * dim2 * 3 * 4 # 3 tensors of 4 bytes each (float32)

    pytorch_bandwidth = bytes_moved / pytorch_mean
    custom_bandwidth = bytes_moved / custom_mean

    # Report
    print(f"Shape: {dim1} x {dim2}")
    print(f"\tPyTorch average execution time: {pytorch_mean*1e3:.3f} milliseconds")
    print(f"\tCustom average execution time: {custom_mean*1e3:.3f} milliseconds")
    print(f"\tPyTorch bandwidth: {pytorch_bandwidth/1e9:.3f} GB/s")
    print(f"\tCustom bandwidth: {custom_bandwidth/1e9:.3f} GB/s")
    if custom_mean < pytorch_mean:
        print(f"\tCustom addition is {(pytorch_mean - custom_mean) / pytorch_mean * 100:.3f}% faster than PyTorch addition")
    else:
        print(f"\tCustom addition is {(custom_mean - pytorch_mean) / pytorch_mean * 100:.3f}% slower than PyTorch addition")


# Run benchmarks for GPT-2 XL addition shapes
shapes = [
    (1024, 768),     # residuals
    (1024, 3072),   # MLP up/down
    (1024, 2304),   # QKV projection bias
]
repetitions = 10000

for dim1, dim2 in shapes:
    benchmark_add(dim1, dim2, repetitions)