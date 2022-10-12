#ifndef TRT_TENSOR_
#define TRT_TENSOR_

#include <string>
#include <vector>
#include <NvInferRuntimeCommon.h>
#include <initializer_list>
#include <cuda.h>
#include "cuda_tools.hpp"

namespace TRT {
    typedef struct { unsigned short _; } float16;
    // 当前数据在哪里
    enum class DataHead {
        Init = 0, // 未分配内存
        Host = 1, // cpu
        Device = 2 // GPU
    };

    enum class DataType {
        Unknow = -1,
        Float = 0,
        Float16 = 1,
        Int32 = 2,
        UInt8 = 3
    };
    //// 数据转移类型
    //// D2H  gpu到cpu
    enum class DataTransType {
        D2H = 0,
        D2D = 1,
        H2H = 2,
        H2D = 3,
    };


    class Tensor {
    public:
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;

        Tensor(DataType dtype = DataType::Float);
        Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float);  // 
        Tensor(const nvinfer1::Dims& dims, DataType dtype = DataType::Float);  // 

        ~Tensor();

        // 赋值
        void set_data(void* data, int byte_size, DataTransType type);

        bool resize(const std::initializer_list<int>& dims);
        bool resize(const std::vector<int>& dims);


        void* cpu() { to_cpu(); return cpu_data_; }
        void* gpu() { to_gpu(); return gpu_data_; }

        template<typename type>
        type* cpu(int i=0) { return (type*)cpu() + i; }

        template<typename type>
        type* gpu(int i=0) { return (type*)gpu() + i; }

        // 维度管理
        inline int  size(int index) { return dims_[index]; };  // 返回指定维度信息
        inline int  batch()         { return dims_[0]; };
        inline int  channel()       { return dims_[1]; };
        inline int  width()         { return dims_[2]; };
        inline int  height()        { return dims_[3]; };
        int         numel();                               // 返回数据个数（非字节数）
        std::vector<int> dims() { return dims_; };
        std::string dim_str();                             // 返回维度的字符串 n x c x h x w

        int byte_size() { return size_; }                  // 返回当前Tensor的字节数

        int data_type_size(DataType dt);


        template<typename type>
        void set_workspace(const std::vector<int>& dims, void* data = nullptr)
        {
            // 分配一大块显存，用于存放原始数据 （预处理之前的）
            int needsize = sizeof(type);
            for (const auto& item : dims)
            {
                needsize *= item;
            }

            if (gpu_workspace_ == nullptr)
            {
                checkCudaRuntime(cudaMalloc(&gpu_workspace_, needsize));
                checkCudaRuntime(cudaMemset(gpu_workspace_, 0, needsize));
                workspace_size_ = needsize;
            }

            if (needsize > workspace_size_)
            {
                // 重新分配内存
                checkCudaRuntime(cudaFree(gpu_workspace_));
                checkCudaRuntime(cudaMalloc(&gpu_workspace_, needsize));
                checkCudaRuntime(cudaMemset(gpu_workspace_, 0, needsize));
                workspace_size_ = needsize;
            }

            if (data != nullptr)
            {
                checkCudaRuntime(cudaMemcpy(gpu_workspace_, data, workspace_size_, cudaMemcpyHostToDevice));
            }
        }

        template<typename type>
        type* get_workspace(int i=0) {  return (type*)gpu_workspace_ + i; }

    private:
        void release_cpu();
        void release_gpu();
        
        void to_cpu();
        void to_gpu();

    private:
        // 此时既要有cpu指针，也要有 gpu指针
        void* cpu_data_{ nullptr };
        void* gpu_data_{nullptr};
        DataHead head_{ DataHead::Init };  // 还未分配空间
        DataType data_type_{ DataType::Unknow };
        int size_{0};  // 目前共用  字节数
        std::vector<int> dims_;   // 该tensor维度  目前要求输出4维tensor

        void* gpu_workspace_{ nullptr };
        int workspace_size_{0};  // 预先分配的大的显存。字节数
    };


};



#endif // !TRT_TENSOR_ 