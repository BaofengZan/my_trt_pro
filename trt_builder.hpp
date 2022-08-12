#pragma once

/*
由onnx模型，编译生成engine
*/
#include <vector>
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

	static std::string join_dims(const std::vector<int>& dims) {
		std::string out;
		for (const auto& item :  dims)
		{
			out += std::to_string(item);
			out += " ";
		}
		return out;
	}

	static std::string mode_str(const Mode& mode) {
		if (mode == Mode::FP32)
		{
			return "fp32";
		}
		else if (mode == Mode::FP16)
		{
			return "fp16";
		}
		else
		{
			return "int8";
		}
	}
	bool compile(
		const Mode& mode,
		unsigned int maxBatchSize,
		const std::string& onnx_file,
		const std::string& engine_save_path,
		const size_t maxWorkspaceSize=1u<<30   // 1G
	);

}; // end namespace TRT
