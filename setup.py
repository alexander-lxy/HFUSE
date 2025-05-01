from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

nvcc_args = ['-O3', '--expt-extended-lambda']
if 'MAX_REG' in os.environ:
    nvcc_args.append('-maxrregcount=' + os.environ['MAX_REG'])

setup(
    name='fusion_cuda',
    ext_modules=[
        CUDAExtension('fusion_cuda', [
            'wrapper.cpp',
            'fuse_kernels/histogram1D_upsample/SummaryUpsample.cu',
        ],
        libraries=['torch'],
        extra_compile_args={'cxx': [],
                           'nvcc': nvcc_args}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })