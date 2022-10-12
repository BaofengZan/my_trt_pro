#ifndef __PREPROCESS_H
#define __PREPROCESS_H


#include <cuda_runtime.h>
#include <cstdint>


struct WarpAffineMatrix{
    float value[6];
};


void preprocess_kernel_img(uint8_t* src, int src_width, int src_height,
                            float* dst, int dst_width, int dst_height, float* out_d2i,
                           cudaStream_t stream);

#endif