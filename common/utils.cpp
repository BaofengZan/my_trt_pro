#include "utils.h"
#include <fstream>
#include <sstream>


#if defined(U_OS_WINDOWS)
#	define HAS_UUID
#	include <Windows.h>
#   include <wingdi.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#   pragma comment(lib, "ole32.lib")
#   pragma comment(lib, "gdi32.lib")
#	undef min
#	undef max
#endif


bool TRT::exists(const std::string& path)
{
#ifdef U_OS_WINDOWS
    return ::PathFileExistsA(path.c_str());
#elif defined(U_OS_LINUX)
    return access(path.c_str(), R_OK) == 0;
#endif
}

bool TRT::mkdir(const std::string& path)
{
#ifdef U_OS_WINDOWS
    return CreateDirectoryA(path.c_str(), nullptr);
#else
    return ::mkdir(path.c_str(), 0755) == 0;
#endif
}

bool TRT::mkdirs(const std::string& path)
{
    if (path.empty()) return false;
    if (exists(path)) return true;

    std::string _path = path;
    char* dir_ptr = (char*)_path.c_str();
    char* iter_ptr = dir_ptr;

    bool keep_going = *iter_ptr != 0;
    while (keep_going) {

        if (*iter_ptr == 0)
            keep_going = false;

#ifdef U_OS_WINDOWS
        if (*iter_ptr == '/' || *iter_ptr == '\\' || *iter_ptr == 0) {
#else
        if ((*iter_ptr == '/' && iter_ptr != dir_ptr) || *iter_ptr == 0) {
#endif
            char old = *iter_ptr;
            *iter_ptr = 0;
            if (!exists(dir_ptr)) {
                if (!mkdir(dir_ptr)) {
                    if (!exists(dir_ptr)) {
                        printf("mkdirs %s return false.", dir_ptr);
                        return false;
                    }
                }
                    }
            *iter_ptr = old;
                }
        iter_ptr++;
            }
    return true;
}


bool TRT::save_file(const std::string& file, const void* data, size_t length, bool mk_dirs)
{
    if (mk_dirs) {
        int p = (int)file.rfind('/');

#ifdef U_OS_WINDOWS
        int e = (int)file.rfind('\\');
        p = std::max(p, e);
#endif
        if (p != -1) {
            if (!mkdirs(file.substr(0, p)))
                return false;
        }
    }

    FILE* f = fopen(file.c_str(), "wb");
    if (!f) return false;

    if (data && length > 0) {
        if (fwrite(data, 1, length, f) != length) {
            fclose(f);
            return false;
        }
    }
    fclose(f);
    return true;
}

std::vector<uint8_t> TRT::load_file(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0) {
        in.seekg(0, std::ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
