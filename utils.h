#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <vector>
#include <string>


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

#endif // !UTILS_HPP_