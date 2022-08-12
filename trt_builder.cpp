#include "trt_builder.hpp"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>


// trt 规定需要继承ILogger，实现logger
// virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;
// AsciiChar const* == const char*
class Logger : public nvinfer1::ILogger {

public:
	virtual void log(Severity severity, const char* msg) {
		if (severity == Severity::kERROR){
			printf("Error: %s\n", msg);   // 这里可以自己使用第三方的logger库，写入到文件中  spd-logger
			abort();
		}
		else if(severity==Severity::kINFO){
			printf("Info: %s\n", msg);
		}
		else if (severity==Severity::kINTERNAL_ERROR) {
			printf("Error: %s\n", msg);
		}
		else if (severity==Severity::kVERBOSE) {
			printf("Verbose: %s\n", msg);
		}
		else{
			//Severity::kWARNING 
			printf("Warning: %s\n", msg);
		}
	}
};

static Logger gLogger;

bool TRT::compile(const Mode& mode, unsigned int maxBatchSize, const std::string& onnx_file, const std::string& engine_save_path, const size_t maxWorkspaceSize)
{
	if (mode == Mode::INT8)
	{
		//INFOE("int8 not emplement.");
		printf("int8 not emplement. \n");
		return false;
	}

	printf("start compile...");

	//自定义删除函数
	std::shared_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger), destroy_nvidia_pointer<nvinfer1::IBuilder>);





	return false;
}
