cmake_minimum_required(VERSION 3.5)

project(my_trt_pro)

add_definitions(-DU_OS_WINDOWS)

# 需要修改的变量
set(OpenCV_DIR "E:\\opencv\\opencv346\\build")
set(TensorRT_DIR "E:\\TensorRT-8.4.1.5.Windows10.x86_64.cuda-10.2.cudnn8.4\\TensorRT-8.4.1.5")


set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

#添加C++11支持及其他选项
set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -g -ftest-coverage -fprofile-arcs")

# cuda
find_package(CUDA REQUIRED)
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIRS})

set(CUDA_NVCC_PLAGS ${CUDA_NVCC_PLAGS};-std=c++11; -g; -G;-gencode; arch=compute_75;code=sm_75)

####
enable_language(CUDA)  # 这一句添加后 ，就会在vs中不需要再手动设置cuda
####

# opencv
# 设置opencv的信息

find_package(OpenCV QUIET
    NO_MODULE
    NO_DEFAULT_PATH
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    NO_SYSTEM_ENVIRONMENT_PATH
    NO_CMAKE_PACKAGE_REGISTRY
    NO_CMAKE_BUILDS_PATH
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_SYSTEM_PACKAGE_REGISTRY
)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})

# nvidia
include_directories(${TensorRT_DIR}\\include)
link_directories(${TensorRT_DIR}\\lib)

# Add sub directories
# 定义子目录，用以递归的调用子目录中的MakeLists.txt
#add_subdirectory(common)
#add_subdirectory(ffhdd)
#add_subdirectory(pipeline)
#add_subdirectory(tensorrt)

# 在目录中查找所有源文件
AUX_SOURCE_DIRECTORY(./common DIR_SRCS)
AUX_SOURCE_DIRECTORY(./ffhdd  DIR_SRCS)
AUX_SOURCE_DIRECTORY(./pipeline DIR_SRCS)
AUX_SOURCE_DIRECTORY(./pipeline/bytetrack/include DIR_SRCS)
AUX_SOURCE_DIRECTORY(./pipeline/bytetrack/src DIR_SRCS)
AUX_SOURCE_DIRECTORY(./tensorrt DIR_SRCS)
AUX_SOURCE_DIRECTORY(./application/yolo DIR_SRCS)

link_directories( ${PROJECT_SOURCE_DIR}/thirdparty/ffmpeg_hx264/ffmpeg-master/lib ${PROJECT_SOURCE_DIR}/thirdparty/Video_Codec_SDK_10.0.26/Lib/x64)

add_executable(${PROJECT_NAME} ${DIR_SRCS} main.cpp)

target_include_directories( ${PROJECT_NAME}
    PUBLIC ${PROJECT_SOURCE_DIR}/thirdparty ${PROJECT_SOURCE_DIR}/thirdparty/spdlog/include ${PROJECT_SOURCE_DIR}/thirdparty/ffmpeg_hx264/ffmpeg-master/include ${PROJECT_SOURCE_DIR}/thirdparty/Video_Codec_SDK_10.0.26/Interface
)

target_link_libraries(my_trt_pro opencv_world)
target_link_libraries(my_trt_pro ${CUDA_LIBRARIES})
target_link_libraries(my_trt_pro nvcuvid nvencodeapi avcodec avfilter avformat swscale swresample avdevice avutil postproc)
target_link_libraries(my_trt_pro nvinfer nvinfer_plugin nvparsers nvonnxparser cudart cuda)