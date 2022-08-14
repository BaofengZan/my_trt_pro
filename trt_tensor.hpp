#pragma one

#include <string>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <initializer_list>

namespace TRT {

    class Tensor {
    public:

        Tensor() = default;
        Tensor(const std::initializer_list<int>& dims);  // 

        ~Tensor();
        bool resize(const std::initializer_list<int>& dims);
        void* gpu() { return data_; }
        int byte_size() { return size_; }
    private:
        void* data_{nullptr};
        int size_{0};
    };

};



