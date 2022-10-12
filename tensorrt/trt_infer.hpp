#ifndef TRT_INFER_
#define TRT_INFER_



#include <string>
#include <vector>
#include <memory>
#include <map>
#include "trt_tensor.hpp"



namespace TRT {

    // �ӿ�  ������
    class Infer {
    public:
        virtual void forward() = 0;
        virtual int get_max_batch_size() = 0;
        virtual int get_input_h() = 0;
        virtual int get_input_w() = 0;
        virtual std::shared_ptr<TRT::Tensor> tensor(const std::string& name) = 0;  // �������� ���ض�Ӧtensor
    };


    std::shared_ptr<Infer> create_engine(const std::string& engine_file);

};


#endif // !TRT_INFER_