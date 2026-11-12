from setuptools import setup, Extension
from torch.utils import cpp_extension

import torch
version = torch.__version__
major = int(version.split('.')[0])
minor = int(version.split('.')[1])

setup(
    name='cedr_backend',
    ext_modules=[cpp_extension.CppExtension(
        'cedr_backend',
        ['cedr_backend.cpp', 'cedr_ops.cpp'],
        include_dirs=[*cpp_extension.include_paths(), 'include'],
        extra_compile_args = ["-fopenmp",
                              f"-DTORCH_MAJOR_VERSION={major}",
                              f"-DTORCH_MINOR_VERSION={minor}",
                              "-DCONV_API=0",
                              ], 
    )],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)

print(f"Built for PyTorch version: {version}")
