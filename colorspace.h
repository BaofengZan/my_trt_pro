#ifndef __COLORSPACE_H
#define __COLORSPACE_H

#include <cuda_runtime.h>
#include <cstdint>

void convert_nv12_to_bgr_invoke(
	const uint8_t* y, const uint8_t* uv, int width, int height, int linesize, uint8_t* dst);

#endif