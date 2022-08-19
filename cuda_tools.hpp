#ifndef CUDA_TOOLS_HPP_
#define CUDA_TOOLS_HPP_
/*
实现一些 cuda的check操作
*/

#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

//namespace CUDATools{
//
//    bool check_driver(CUresult e, const char* call, int line, const char* file) {
//        if (e != CUDA_SUCCESS) {
//
//            const char* message = nullptr;
//            const char* name = nullptr;
//            cuGetErrorString(e, &message);
//            cuGetErrorName(e, &name);
//            printf("CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
//            return false;
//        }
//        return true;
//    }
//
//
//    bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
//        if (e != cudaSuccess) {
//            printf("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
//            return false;
//        }
//        return true;
//    }
//
//
//};
//
// 

#define GPU_BLOCK_THREADS  512


#define KernelPositionBlock											\
	int position = (blockDim.x * blockIdx.x + threadIdx.x);		    \
    if (position >= (edge)) return;

#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        printf("launch failed: %s", cudaGetErrorString(cudaStatus));                                  \
    }} while(0);



namespace CUDATools{

    bool check_driver(CUresult e, const char* call, int line, const char* file);


    bool check_runtime(cudaError_t e, const char* call, int line, const char* file);

    dim3 grid_dims(int numJobs);
    dim3 block_dims(int numJobs);
};


#define checkCudaDriver(call)  CUDATools::check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) CUDATools::check_runtime(call, #call, __LINE__, __FILE__)

//#define checkCudaDriver(call)  
//#define checkCudaRuntime(call)


#endif // !CUDA_TOOLS_HPP_