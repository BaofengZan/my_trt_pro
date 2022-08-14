#pragma once
/*
实现一些 cuda的check操作
*/

#include <cuda.h>
#include <cuda_runtime.h>


#define checkCudaDriver(call)  CUDATools::check_driver(call, #call, __LINE__, __FILE__)
#define checkCudaRuntime(call) CUDATools::check_runtime(call, #call, __LINE__, __FILE__)


namespace CUDATools {

    bool check_driver(CUresult e, const char* call, int line, const char* file) {
        if (e != CUDA_SUCCESS) {

            const char* message = nullptr;
            const char* name = nullptr;
            cuGetErrorString(e, &message);
            cuGetErrorName(e, &name);
            printf("CUDA Driver error %s # %s, code = %s [ %d ] in file %s:%d", call, message, name, e, file, line);
            return false;
        }
        return true;
    }


    bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
        if (e != cudaSuccess) {
            printf("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }


};

