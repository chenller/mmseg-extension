# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

import glob
import os
from pprint import pprint

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

requirements = ['torch', 'torchvision']


def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, 'mmsegextlib_msda/src')

    main_file = glob.glob(os.path.join(extensions_dir, '*.cpp'))
    source_cpu = glob.glob(os.path.join(extensions_dir, 'cpu', '*.cpp'))
    source_cuda = glob.glob(os.path.join(extensions_dir, 'cuda', '*.cu'))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {'cxx': []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [('WITH_CUDA', None)]
        extra_compile_args['nvcc'] = [
            '-DCUDA_HAS_FP16=1',
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            'mmsegextlib_msda',
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    # tmp = dict(
    #     a='mmsegextlib_MSDA',
    #     s=sources,
    #     include_dirs=include_dirs,
    #     define_macros=define_macros,
    #     extra_compile_args=extra_compile_args,
    # )
    # pprint(tmp)
    return ext_modules


# get_extensions()
setup(
    name='mmsegextlib_msda',
    version='1.0',
    author='Weijie Su',
    url='https://github.com/fundamentalvision/Deformable-DETR',
    description=
    'PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention',
    packages=find_packages(),
    ext_modules=get_extensions(),
    cmdclass={'build_ext': torch.utils.cpp_extension.BuildExtension},
)
