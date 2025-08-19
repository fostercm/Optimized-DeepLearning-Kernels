from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='test_extension',
    ext_modules=[
        CUDAExtension(
            name='test_extension',
            sources=['addition/addition.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)