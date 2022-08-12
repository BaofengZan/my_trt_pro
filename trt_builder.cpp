#include "trt_builder.hpp"
#include "utils.h"
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
	virtual void log(Severity severity, const char* msg) noexcept {
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
	if (builder == nullptr){
		printf("create builer error!\n");
		return false;
	}

	std::shared_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<nvinfer1::IBuilderConfig>);

	if (mode == Mode::FP16){
		if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		else
		{
			printf("Plateform not support fp16\n");
		}
	}
	else if (mode == Mode::INT8) {
		if (builder->platformHasFastInt8())
		{
			config->setFlag(nvinfer1::BuilderFlag::kINT8);
		}
		else
		{
			printf("Plateform not support int 8\n");
		}
	}

	// 开始parse onnx
	std::shared_ptr<nvinfer1::INetworkDefinition> network;

	// 先支持onnx
	const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	network = std::shared_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch), destroy_nvidia_pointer<nvinfer1::INetworkDefinition>);

	// 解析onnx
	std::shared_ptr<nvonnxparser::IParser> onnxParser;

	//！ 这里在Tensorrt_pro中该接口有修改
	onnxParser = std::shared_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger), destroy_nvidia_pointer<nvonnxparser::IParser>);
	if (!onnxParser->parseFromFile(onnx_file.c_str(), 1)) // 1 详细打印日志
	{
		printf("parse onnx file error!\n");
		return false;
	}

	// 打印一些输入 输出信息
	auto inputTensor = network->getInput(0);
	auto inputDims = inputTensor->getDimensions();

	printf("Input shape is %s\n", join_dims(std::vector<int>(inputDims.d, inputDims.d+inputDims.nbDims)).c_str());

	int num_input = network->getNbInputs();
	printf("Network has %d inputs\n", num_input);
	for (int i = 0; i < num_input; ++i) {
		auto tensor = network->getInput(i);
		auto dims = tensor->getDimensions();
		auto dim_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
		printf("input[%d] %s shape is %s \n", i, tensor->getName(), dim_str.c_str());
	}

	int num_output = network->getNbOutputs();
	printf("Network has %d outputs\n", num_output);
	for (int i = 0; i < num_output; ++i)
	{
		auto tensor = network->getOutput(i);
		auto dims = tensor->getDimensions();
		auto dim_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
		printf("input[%d] %s shape is %s \n", i, tensor->getName(), dim_str.c_str());
	}

	// 打印每一层的维度
	int num_net_layers = network->getNbLayers();
	printf("Network has %d layers\n", num_net_layers);
	for (int i = 0; i < num_net_layers; ++i)
	{
		auto layer = network->getLayer(i);
		auto name = layer->getName();
		//auto input0 = layer->getInput(0);
		auto input = layer->getNbInputs();
		std::string input_dims = "";
		for (int j=0; j<input; ++j)
		{
			auto dims = layer->getInput(j)->getDimensions();
			input_dims += "[";
			input_dims += join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
			input_dims += "]";
		}

		auto output = layer->getNbOutputs();
		std::string out_dims = "";
		for (int j = 0; j < output; ++j)
		{
			auto dims = layer->getOutput(j)->getDimensions();
			out_dims += "[";
			out_dims += join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
			out_dims += "]";
		}

		printf("[%s]  %s --> %s \n", name, input_dims.c_str(), out_dims.c_str());
	}

	builder->setMaxBatchSize(maxBatchSize);
	printf("set MaxBatchSize = %d\n", maxBatchSize);

	config->setMaxWorkspaceSize(maxWorkspaceSize);
	printf("set MaxBatchSize = %.2fMB\n", maxBatchSize/1024.0f/1024.0f);

	// profile
	// 为不同输入设置不同大小
	auto profile = builder->createOptimizationProfile();
	for (int i = 0; i < num_input; ++i)
	{
		auto input = network->getInput(i);
		auto input_dim = input->getDimensions();
		input_dim.d[0] = 1;
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dim);
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dim);
		input_dim.d[0] = maxBatchSize;
		profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dim);
	}

	config->addOptimizationProfile(profile);

	printf("\n\n\nBuild Engine.....\n");
	std::shared_ptr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
	if (engine == nullptr)
	{
		printf("engine is nullptr\n");
		return false;
	}

	std::shared_ptr<nvinfer1::IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<nvinfer1::IHostMemory>);

	//save file
	size_t index = 0;
	if (onnx_file.find("/") != std::string::npos)
	{
		index = onnx_file.find_last_of("/");
	}
	else if (onnx_file.find("\\") != std::string::npos)
	{
		index = onnx_file.find_last_of("\\");
	}
	std::string mode_name = "";
	if (index != 0)
	{
		mode_name = onnx_file.substr(index + 1);
		mode_name.replace(mode_name.find_last_of("."), 4, ".engine");
	}
	else
	{
		mode_name = "engine.engine";
	}
	std::string engine_file = engine_save_path + "/" + mode_str(mode)+ "_" + mode_name;
	printf("Save engine to %s\n", engine_file.c_str());
	return save_file(engine_file, seridata->data(), seridata->size());
}
