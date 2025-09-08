import torch
import torch.utils.benchmark as benchmark
import fastkern as fk

def benchmark_mult(dim1: int, dim2: int, dim3: int, repetitions: int) -> None:
    a = torch.randn((dim1, dim2), dtype=torch.float32, device='cuda')
    b = torch.randn((dim2, dim3), dtype=torch.float32, device='cuda')

    # Warm up
    for _ in range(10):
        _ = a @ b
        _ = fk.mult(a, b)
    torch.cuda.synchronize()

    # PyTorch Multiplication
    pytorch_timer = benchmark.Timer(
        stmt=
        '''
        c = a @ b
        ''',
        globals={'a': a, 'b': b}
    )
    
    # Custom Multiplication
    custom_timer = benchmark.Timer(
        stmt=
        '''
        c = fk.mult(a, b)
        ''',
        globals={'a': a, 'b': b, 'fk': fk}
    )

    # Measure time and bandwidth
    pytorch_result = pytorch_timer.timeit(repetitions)
    custom_result = custom_timer.timeit(repetitions)

    pytorch_mean = pytorch_result.mean
    custom_mean = custom_result.mean

    bytes_moved = (dim1 * dim2 + dim2 * dim3 + dim1 * dim3) * 4 # 3 tensors of 4 bytes each (float32)

    pytorch_bandwidth = bytes_moved / pytorch_mean
    custom_bandwidth = bytes_moved / custom_mean

    # Report
    print(f"Shape: {dim1} x {dim2} @ {dim2} x {dim3}")
    print(f"\tPyTorch average execution time: {pytorch_mean*1e3:.3f} milliseconds")
    print(f"\tCustom average execution time: {custom_mean*1e3:.3f} milliseconds")
    print(f"\tPyTorch bandwidth: {pytorch_bandwidth/1e9:.3f} GB/s")
    print(f"\tCustom bandwidth: {custom_bandwidth/1e9:.3f} GB/s")
    if custom_mean < pytorch_mean:
        print(f"\tCustom multiplication is {(pytorch_mean - custom_mean) / pytorch_mean * 100:.3f}% faster than PyTorch multiplication")
    else:
        print(f"\tCustom multiplication is {(custom_mean - pytorch_mean) / pytorch_mean * 100:.3f}% slower than PyTorch multiplication")
    
# Run benchmarks for GPT-2 XL multiplication shapes
shapes = [
    (1024, 1600, 6400),   # MLP up-projection
    (1024, 6400, 1600),   # MLP down-projection
    (1024, 64, 1024)      # Attention scoring
]
repetitions = 100

for dim1, dim2, dim3 in shapes:
    benchmark_mult(dim1, dim2, dim3, repetitions)