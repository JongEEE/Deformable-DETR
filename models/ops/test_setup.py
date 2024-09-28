import os
import glob

import torch

from torch.utils.cpp_extension import CUDA_HOME
from torch.utils.cpp_extension import CppExtension
from torch.utils.cpp_extension import CUDAExtension

# 测试 setup 文件中的几个函数

def test():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # print("this_dir", this_dir)
    extensions_dir = os.path.join(this_dir, "src")
    # print("extensions_dir", extensions_dir)

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))
    # for file in main_file:
    #     print("file: ", file)

    sources = main_file + source_cpu

    # for src_file in sources:
    #     print("srcFile:", src_file)
    
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    # 条件编译
    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]
    else:
        raise NotImplementedError('Cuda is not availabel')
    
    # print(CUDA_HOME)
    
    sources = [os.path.join(extensions_dir, s) for s in sources]
    # for src_file in sources:
    #     print("srcFile:", src_file)
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    print(ext_modules)
    return ext_modules



if __name__ == "__main__":
    test()