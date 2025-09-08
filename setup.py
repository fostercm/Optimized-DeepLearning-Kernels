from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

ext_modules=[
    CUDAExtension(
        name="fastkern._add",
        sources=[
            "fastkern/add/add.cu"
        ]
    ),
    CUDAExtension(
        name="fastkern._mult",
        sources=[
            "fastkern/mult/mult.cu"
        ]
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