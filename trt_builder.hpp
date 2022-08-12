#pragma once

/*
由onnx模型，编译生成engine
*/


#include <string>
#include <vector>


namespace TRT {
	enum class Mode : int {
		FP32,
		FP16,
		INT8
	};

	template<typename _T>
	static void destroy_nvidia_pointer(_T* ptr) {
		if (ptr) ptr->destroy();
	}

	bool compile(
		const Mode& mode,
		unsigned int maxBatchSize,
		const std::string& onnx_file,
		const std::string& engine_save_path,
		const size_t maxWorkspaceSize=1u<<30   // 1G
	);

}; // end namespace TRT
