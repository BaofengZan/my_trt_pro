#ifndef TRT_INFER_
#define TRT_INFER_



#include <string>
#include <vector>
#include <memory>
#include <map>
#include "trt_tensor.hpp"



namespace TRT {

    // 接口  纯虚类
    class Infer {
    public:
        virtual void forward() = 0;
        virtual int get_max_batch_size() = 0;
        virtual int get_input_h() = 0;
        virtual int get_input_w() = 0;
        virtual std::shared_ptr<TRT::Tensor> tensor(const std::string& name) = 0;  // 根据名字 返回对应tensor
    };


    std::shared_ptr<Infer> create_engine(const std::string& engine_file);

};


#endif // !TRT_INFER_