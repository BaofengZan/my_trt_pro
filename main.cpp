#include <iostream>
#include "trt_builder.hpp"

int main()
{


	TRT::compile(
		TRT::Mode::FP16, 
		1, 
		R"(F:\LearningCode\tensorRT_Pro\workspace\yolov5s.onnx)",
		R"(F:\LearningCode\tensorRT_Pro\workspace)"
	);
	return 0;
}