#include "trt_tensor.hpp"
#include "cuda_tools.hpp"
#include <cuda.h>

//TRT::Tensor::Tensor(int size): size_(size)
//{
//    checkCudaRuntime(cudaMalloc(&data_, size));
//}

TRT::Tensor::Tensor(const std::initializer_list<int>& dims)
{
    resize(dims);
}

TRT::Tensor::~Tensor()
{
    checkCudaRuntime(cudaFree(data_));
}

bool TRT::Tensor::resize(const std::initializer_list<int>& dims)
{
    int byte_size = sizeof(float);
    for (auto& item: dims)
    {
        byte_size *= item;
    }

    checkCudaRuntime(cudaMalloc(&data_, byte_size));
}



