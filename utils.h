#pragma once
#include <vector>
#include <string>


namespace TRT
{
    bool exists(const std::string& path);
    bool mkdir(const std::string& path);
    bool mkdirs(const std::string& path);
    bool save_file(const std::string& file, const void* data, size_t length, bool mk_dirs = true);
}
