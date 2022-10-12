#include "../common/cuda_tools.hpp"
#include "trt_tensor.hpp"
#include <cuda.h>
#include <cuda_fp16.h>
#include <assert.h>
#include "log.h"
//TRT::Tensor::Tensor(int size): size_(size)
//{
//    checkCudaRuntime(cudaMalloc(&data_, size));
//}

TRT::Tensor::Tensor(DataType dtype)
{
    data_type_ = dtype;
}

TRT::Tensor::Tensor(const std::vector<int>& dims, DataType dtype)
{
    dims_ = dims;
    data_type_ = dtype;
    resize(dims);
}

TRT::Tensor::Tensor(const nvinfer1::Dims& dims, DataType dtype)
{
    data_type_ = dtype;
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
    release_cpu();
    release_gpu();
}


void TRT::Tensor::set_data(void* data, int byte_size, DataTransType type)
{
    assert(byte_size <= size_);
    if (type == DataTransType::D2D)
    {
        // 直接调用，gpu()中会先分配显存的
        cudaMemcpy(gpu(), data, byte_size, cudaMemcpyDeviceToDevice);
    }
    else if (type == DataTransType::H2D)
    {
        // 直接调用，gpu()中会先分配显存的
        cudaMemcpy(gpu(), data, byte_size, cudaMemcpyHostToDevice);
    }
    else if (type == DataTransType::H2H)
    {
        // 直接调用，gpu()中会先分配显存的
        memcpy(cpu(), data, byte_size);
    }
    else if (type == DataTransType::D2H)
    {
        // 直接调用，gpu()中会先分配显存的
        cudaMemcpy(cpu(), data, byte_size, cudaMemcpyDeviceToHost);
    }

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
    //checkCudaRuntime(cudaMalloc(&data_, byte_size));
    return true;
}

bool TRT::Tensor::resize(const std::vector<int>& dims)
{
    dims_ = dims;
    int byte_size =  data_type_size(data_type_);
    for (auto& item : dims)
    {
        byte_size *= item;
    }
    if (byte_size > size_)
    {
        head_ = DataHead::Init;
    }
    size_ = byte_size;
    //checkCudaRuntime(cudaMalloc(&data_, byte_size));
    return true;
}

int TRT::Tensor::numel()
{
    int c = dims_.empty() ? 0 : 1;
    for (const auto& item : dims_)
    {
        c *= item;
    }
    return c;
}

std::string TRT::Tensor::dim_str()
{
    if (dims_.empty())
    {
        return "";
    }
    std::string s="";
    for (int i=0; i<dims_.size(); ++i)
    {
        s += std::to_string(dims_[i]);
        if (i < dims_.size() - 1)
        {
            s += " x ";
        }
    }
    return s;
}

int TRT::Tensor::data_type_size(DataType dt) {
    switch (dt) {
    case DataType::Float: return sizeof(float);
    case DataType::Float16: return sizeof(float16); //     typedef struct{unsigned short _;} float16;
    case DataType::Int32: return sizeof(int);
    case DataType::UInt8: return sizeof(uint8_t);
    default: {
        spdlog::error("Not support dtype");
        return -1;
    }
    }
}

void TRT::Tensor::release_cpu()
{
    if (cpu_data_)
    {
        checkCudaRuntime(cudaFreeHost(cpu_data_));
        cpu_data_ = nullptr;
    }
}

void TRT::Tensor::release_gpu()
{
    if (gpu_data_)
    {
        checkCudaRuntime(cudaFree(gpu_data_));
        gpu_data_ = nullptr;
    }
    if (gpu_workspace_)
    {
        checkCudaRuntime(cudaFree(gpu_workspace_));
        gpu_workspace_ = nullptr;
    }
}

void TRT::Tensor::to_cpu()
{
    if (head_ == DataHead::Host)
    {
        // 不用转移。
        return;
    }
    head_ = DataHead::Host;

    // 开始分配内存
    checkCudaRuntime(cudaMallocHost(&cpu_data_, size_));
    memset(cpu_data_, 0, size_);

    // 将gpu内存拷贝到cpu
    if (gpu_data_ != nullptr)
    {
        checkCudaRuntime(cudaMemcpy(cpu_data_, gpu_data_, size_, cudaMemcpyDeviceToHost));
    }
    return;
}

void TRT::Tensor::to_gpu()
{
    if (head_ == DataHead::Device)
    {
        // 不用转移。
        return;
    }
    head_ = DataHead::Device;

    // 开始分配内存
    checkCudaRuntime(cudaMalloc(&gpu_data_, size_));
    cudaMemset(gpu_data_, 0, size_);

    // 将cpu内存拷贝到gpu
    if (cpu_data_ != nullptr)
    {
        checkCudaRuntime(cudaMemcpy(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice));
    }
    return;
}



