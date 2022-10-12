#include <iostream>
#include <queue>
#include <memory>
#include "trt_builder.hpp"
#include "trt_infer.hpp"
#include "yolo.hpp"
#include "./pipeline/AsyncProcess2Threads.h"
#include "opencv2/opencv.hpp"
#include "log.h"
#include "ffhdd/ffmpeg_demuxer.hpp"
#include "ffhdd/cuvid_decoder.hpp"
#include "ffhdd/nalu.hpp"


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

void test_decode() {
    //测试 硬解码
    // 1 测试是否能正常解码
    auto demuxer = FFHDDemuxer::create_ffmpeg_demuxer("F:\\LearningCode\\hard_decode_trt-windows\\workspace\\exp/face_tracker.mp4");
    if (demuxer == nullptr) {
        return;
    }

    auto decoder = FFHDDecoder::create_cuvid_decoder(
        false, FFHDDecoder::ffmpeg2NvCodecId(demuxer->get_video_codec()), -1, 0
    );

    if (decoder == nullptr) {
        return;
    }

    uint8_t* packet_data = nullptr;
    int packet_size = 0;
    int64_t pts = 0;

    demuxer->get_extra_data(&packet_data, &packet_size);
    decoder->decode(packet_data, packet_size);


    do {
        demuxer->demux(&packet_data, &packet_size, &pts);
        int ndecoded_frame = decoder->decode(packet_data, packet_size, pts);
        for (int i = 0; i < ndecoded_frame; ++i) {
            unsigned int frame_index = 0;

            /* 因为decoder获取的frame内存，是YUV-NV12格式的。储存内存大小是 [height * 1.5] * width byte
             因此构造一个height * 1.5,  width 大小的空间
             然后由opencv函数，把YUV-NV12转换到BGR，转换后的image则是正常的height, width, CV_8UC3
            */
            cv::Mat image(decoder->get_height() * 1.5, decoder->get_width(), CV_8U, decoder->get_frame(&pts, &frame_index));
            cv::cvtColor(image, image, cv::COLOR_YUV2BGR_NV12);

            frame_index = frame_index + 1;
            //cv::imwrite(cv::format("imgs/img_%05d.jpg", frame_index), image);
            cv::imshow("11", image);
            cv::waitKey(0);
        }
    } while (packet_size > 0);
}


void test_yolo()
{
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
        float b[4] = { (obj.left + obj.right) / 2 , (obj.top + obj.bottom) / 2, (obj.right - obj.left), (obj.bottom - obj.top) };
        auto box = get_rect(img, b);
        cv::rectangle(img, box, cv::Scalar(0, 0, 255), 5);
        // cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);
    }

    cv::imwrite(R"(F:\LearningCode\tensorRT_Pro\workspace\inference\car_det.jpg)", img);
}


void test_Async(){
    // 测试多路
    int yolo_device_id = 0;

    int num_yolo_instance = 1;
    int num_videos = 2;
    std::vector<std::shared_ptr<Yolo::Infer>> yolo_instance;
    for (int i = 0; i < num_yolo_instance; ++i)
    {
        auto yolo = Yolo::create_infer(R"(F:\LearningCode\my_trt_pro\my_trt_pro/fp16_yolov5s-err.engine)", 0, 0.45, 0.5);;
        if (yolo == nullptr) {
            std::cout << "Yolo create failed \n";
            continue;
        }
        yolo_instance.emplace_back(std::move(yolo));
    }

    // warm up
    for (size_t i = 0; i < yolo_instance.size(); ++i)
    {
        for (int j = 0; j < 3; ++j) {
            yolo_instance[i]->commit(cv::Mat(640, 640, CV_8UC3)).get();
        }
    }


    std::string file = "F:\\LearningCode\\hard_decode_trt-windows\\workspace\\exp/face_tracker.mp4";
    std::vector<std::thread> ts;


    auto func = [&](std::shared_ptr<Yolo::Infer>& yolo_, std::string in_file, int id) {
        AsyncProcess2Threads async(yolo_, in_file, id);
        async.AsyncProcess();
    };

    for (int i = 0; i < num_videos; ++i) {
        ts.emplace_back(std::bind(func, std::ref(yolo_instance[i % yolo_instance.size()]), file, i));
    }

    for (auto& t : ts) {
        t.join();
    }

}
int main()
{

    init_logger(); // ok

    //test_yolo(); 

    //test_decode();  // 测试ok

    test_Async();
    
	return 0;
}