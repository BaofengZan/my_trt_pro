#ifndef YOLO_HPP_
#define YOLO_HPP_

#include <vector>
#include <memory>
#include <string>
#include <future>

#include "opencv2/opencv.hpp"
#include "utils.h"
#include "trt_tensor.hpp"

//namespace cv
//{
//    class Mat;
//};

// v5
namespace Yolo {
    using namespace ObjectBox;


    class Infer {
        // 定义接口。
        // 具体的实现再Impl中
    public:
        virtual std::shared_future<BoxArray> commit(const Yolo::Image& image) = 0;
        //virtual std::shared_future<BoxArray> commit_gpu(const Yolo::Image& image) = 0;
        //virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) = 0;
    };


    std::shared_ptr<Infer> create_infer(
        const std::string& engine_file,
        int gpuid = 0,
        float confidence_threshold = 0.3f,
        float nms_threshold = 0.5f,
        int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );
};

#endif // !YOLO_HPP_