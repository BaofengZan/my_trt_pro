#ifndef TRT_TENSOR_
#define TRT_TENSOR_

#include <string>
#include <vector>
#include <NvInferRuntimeCommon.h>
#include <initializer_list>

namespace TRT {

    class Tensor {
    public:

        Tensor() = default;
        Tensor(const std::initializer_list<int>& dims);  // 
        Tensor(const nvinfer1::Dims& dims);  // 

        ~Tensor();
        bool resize(const std::initializer_list<int>& dims);
        bool resize(const std::vector<int>& dims);
        void* gpu() { return data_; }
        int byte_size() { return size_; }

        int size(int index) { return dims_[index]; };
    private:
        void* data_{nullptr};
        int size_{0};
        std::vector<int> dims_;   // ¸ÃtensorÎ¬¶È
    };

};



#endif // !TRT_TENSOR_