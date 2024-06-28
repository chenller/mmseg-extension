'''
TransNeXt: Robust Foveal Visual Perception for Vision Transformers
Paper: https://arxiv.org/abs/2311.17132
Code: https://github.com/DaiShiResearch/TransNeXt

Author: Dai Shi
Github: https://github.com/DaiShiResearch
Email: daishiresearch@gmail.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''

import glob
from pprint import pprint
import os.path as osp
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('**/*.cpp') + glob.glob('**/*.cu')
print('include file:')
pprint(include_dirs)
print('sources file:')
pprint(sources)

setup(
    name='mmsegextlib_swattention',
    version='1.0',
    author='daishi',
    author_email='daishiresearch@gmail.com',
    description='swattention',
    long_description='swattention',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='swattention',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
