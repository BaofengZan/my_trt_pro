#include <iostream>
#include "trt_builder.hpp"
#include "trt_infer.hpp"
#include "yolo.hpp"
#include <queue>
#include "opencv2/opencv.hpp"
#include "log.h"

namespace Yolo
{
    const int INPUT_W = 640;
    const int INPUT_H = 640;
};

// 传进来的box是cx cy w h
cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    float l, r, t, b; 
    float r_w = Yolo::INPUT_W / (img.cols * 1.0);
    float r_h = Yolo::INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2] / 2.f;
        r = bbox[0] + bbox[2] / 2.f;
        t = bbox[1] - bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3] / 2.f - (Yolo::INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    }
    else {
        l = bbox[0] - bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2] / 2.f - (Yolo::INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3] / 2.f;
        b = bbox[1] + bbox[3] / 2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(round(l), round(t), round(r - l), round(b - t));
}

int main()
{

    init_logger();

	//TRT::compile(
	//	TRT::Mode::FP16, 
	//	1, 
	//	R"(F:\LearningCode\my_trt_pro\my_trt_pro\yolov5s-err.onnx)",
	//	R"(F:\LearningCode\my_trt_pro\my_trt_pro)"
	//);

	// 开始编写infer
	// 利用封装的思想
		// 解析engine文件，生成engine
		// job管理。
		// tensor管理

	//auto infer = TRT::create_engine(R"(F:\LearningCode\tensorRT_Pro\workspace/fp16_yolov5s.engine)");
	//infer->forward();
    //gpuid =0, confidence_threshold=0.45, nms_threshold=0.5
	auto engine = Yolo::create_infer(R"(F:\LearningCode\my_trt_pro\my_trt_pro/fp16_yolov5s-err.engine)", 0, 0.45, 0.5);
	cv::Mat img = cv::imread(R"(F:\LearningCode\tensorRT_Pro\workspace\inference\car.jpg)");
	std::shared_future<Yolo::BoxArray> predbox = engine->commit(img);

	auto boxes = predbox.get();

	for (auto& obj : boxes) {
        float b[4] = { (obj.left + obj.right)/2 , (obj.top+obj.bottom)/2, (obj.right - obj.left), (obj.bottom-obj.top)};
        auto box = get_rect(img, b);
		cv::rectangle(img, box, cv::Scalar(0, 0, 255), 5);
        // cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
	}

	cv::imwrite(R"(F:\LearningCode\tensorRT_Pro\workspace\inference\car_det.jpg)", img);
	return 0;
}