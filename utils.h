#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

namespace TRT
{
    bool exists(const std::string& path);
    bool mkdir(const std::string& path);
    bool mkdirs(const std::string& path);
    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);
    std::vector<uint8_t> load_file(const std::string& file);
}

namespace ObjectBox {

    struct Box {
        float left, top, right, bottom, confidence;
        int class_label;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
            :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label) {}
    };

    typedef std::vector<Box> BoxArray;
};



namespace Yolo {
    enum class Type : int {
        V5 = 0,
        X = 1
    };

    enum class NMSMethod : int {
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    enum class ImageType : int {
        CVMat = 0,
        GPUYUV = 1    // nv12
    };

    struct Image {
        ImageType type = ImageType::CVMat;
        cv::Mat cvmat;   // 用于画图

        // GPU YUV image
        //TRT::CUStream stream = nullptr;  // 目前先支持 默认流
        uint8_t* device_data = nullptr;
        int width = 0, height = 0;
        int device_id = 0;
        Image() = default;
        ~Image()
        {
        }
        Image(const cv::Mat& cvmat) :cvmat(cvmat), type(ImageType::CVMat) {}
        Image(uint8_t* yuvdata_device, int width, int height, int device_id)
            :device_data(yuvdata_device), width(width), height(height), device_id(device_id), type(ImageType::GPUYUV){
            cvmat = cv::Mat(height, width, CV_8UC3, cv::Scalar{ 0 });
        }

        int get_width() const { return type == ImageType::CVMat ? cvmat.cols : width; }
        int get_height() const { return type == ImageType::CVMat ? cvmat.rows : height; }
        cv::Size get_size() const { return cv::Size(get_width(), get_height()); }
        bool empty() const { return type == ImageType::CVMat ? cvmat.empty() : (device_data == nullptr || width < 1 || height < 1); }
    };

};

#endif // !UTILS_HPP_