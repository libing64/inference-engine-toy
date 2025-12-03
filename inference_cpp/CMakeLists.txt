cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(inference_cpp)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找 LibTorch
# 方法1: 如果 LibTorch 已安装在系统路径
# find_package(Torch REQUIRED)

# 方法2: 从环境变量获取 LibTorch 路径（推荐）
# 使用前需要设置: export Torch_DIR=/path/to/libtorch/share/cmake/Torch
if(DEFINED ENV{Torch_DIR})
    set(Torch_DIR $ENV{Torch_DIR})
    find_package(Torch REQUIRED)
else()
    # 方法3: 手动指定 LibTorch 路径（请根据实际情况修改）
    # set(Torch_DIR "/path/to/libtorch/share/cmake/Torch")
    # find_package(Torch REQUIRED)
    
    message(FATAL_ERROR 
        "未找到 LibTorch。请选择以下方式之一:\n"
        "1. 设置环境变量: export Torch_DIR=/path/to/libtorch/share/cmake/Torch\n"
        "2. 在 CMakeLists.txt 中手动设置 Torch_DIR 路径\n"
        "3. 使用系统安装的 LibTorch: find_package(Torch REQUIRED)\n\n"
        "下载 LibTorch: https://pytorch.org/get-started/locally/"
    )
endif()

# 添加可执行文件
add_executable(inference inference.cpp)

# 链接 LibTorch
target_link_libraries(inference "${TORCH_LIBRARIES}")

# 设置编译选项
set_property(TARGET inference PROPERTY CXX_STANDARD 17)

# 打印信息
message(STATUS "LibTorch 版本: ${TORCH_VERSION}")
message(STATUS "LibTorch 库路径: ${TORCH_LIBRARIES}")
