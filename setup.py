from setuptools import setup, find_packages
from torch.cuda import get_device_capability
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

arch = '.'.join(map(str, get_device_capability(0)))
os.environ['TORCH_CUDA_ARCH_LIST'] = arch

extra_compile_args = {
    "nvcc": ["-O3", "-g", "-lineinfo"]
}

ext_modules=[
    CUDAExtension(
        name="fastkern._add",
        sources=[
            "fastkern/add/add.cu"
        ],
        extra_compile_args=extra_compile_args
    ),
    CUDAExtension(
        name="fastkern._mult",
        sources=[
            "fastkern/mult/mult.cu"
        ],
        extra_compile_args=extra_compile_args
    )
]

setup(
    name='fastkern',
    version='0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    },
    zip_safe=False
)