from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules=[
    CUDAExtension(
        name="fastkern._add",
        sources=[
            "fastkern/add/add.cu"
        ]
    )
]

setup(
    name='fastkern',
    version='0.1',
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    zip_safe=False
)