#include "cuda_tools.hpp"
#include "log.h"

namespace CUDATools
{
        bool check_driver(CUresult e, const char* call, int line, const char* file) {
           if (e != CUDA_SUCCESS) {
   
               const char* message = nullptr;
               const char* name = nullptr;
               cuGetErrorString(e, &message);
               cuGetErrorName(e, &name);
               spdlog::info("CUDA Driver error {} # {}, code = {} [ {} ] in file {}:{}", call, message, name, e, file, line);
               return false;
           }
           return true;
       }
   
   
       bool check_runtime(cudaError_t e, const char* call, int line, const char* file) {
           if (e != cudaSuccess) {
               spdlog::info("CUDA Runtime error {} # {}, code = {} [ {} ] in file {}:{}", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
               return false;
           }
           return true;
       }
    dim3 grid_dims(int numJobs) {
        int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
        return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
    }

    dim3 block_dims(int numJobs) {
        return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    }


    AutoDevice::AutoDevice(int device_id)
    {
        cudaGetDevice(&old_);
        if (old_ != device_id && device_id != -1) {
            checkCudaRuntime(cudaSetDevice(device_id));
            return;
        }

        CUcontext context = nullptr;
        cuCtxGetCurrent(&context);
        if (context == nullptr) {
            checkCudaRuntime(cudaSetDevice(device_id));
            return;
        }
    }
    AutoDevice::~AutoDevice() {
        if (old_ != -1) {
            checkCudaRuntime(cudaSetDevice(old_));
        }
    }
}

