#include "cuda_tools.hpp"
#include "trt_tensor.hpp"
#include <cuda.h>

//TRT::Tensor::Tensor(int size): size_(size)
//{
//    checkCudaRuntime(cudaMalloc(&data_, size));
//}

TRT::Tensor::Tensor(const std::initializer_list<int>& dims)
{
    dims_ = dims;
    resize(dims);
}

TRT::Tensor::Tensor(const nvinfer1::Dims& dims)
{
    std::vector<int> d;
    for (int i = 0; i < dims.nbDims; i++)
    {
        d.push_back(dims.d[i]);
    }
    dims_ = d;
    resize(d);
}

TRT::Tensor::~Tensor()
{
    checkCudaRuntime(cudaFree(data_));
}

bool TRT::Tensor::resize(const std::initializer_list<int>& dims)
{
    dims_ = dims;
    int byte_size = sizeof(float);
    for (auto& item: dims)
    {
        byte_size *= item;
    }
    size_ = byte_size;
    checkCudaRuntime(cudaMalloc(&data_, byte_size));
    return true;
}

bool TRT::Tensor::resize(const std::vector<int>& dims)
{
    dims_ = dims;
    int byte_size = sizeof(float);
    for (auto& item : dims)
    {
        byte_size *= item;
    }
    size_ = byte_size;
    checkCudaRuntime(cudaMalloc(&data_, byte_size));
    return true;
}



