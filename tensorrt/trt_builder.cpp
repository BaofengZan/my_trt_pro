#include "trt_builder.hpp"
#include "trt_logging.hpp"
#include "../common/utils.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include "log.h"

// trt 规定需要继承ILogger，实现logger
// virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;
// AsciiChar const* == const char*
// 
//class Logger : public nvinfer1::ILogger {
//	// critical > error > warn > info > debug > trace
//public:
//	virtual void log(Severity severity, const char* msg) noexcept {
//		if (severity == Severity::kERROR) {
//			spdlog::critical("Error: {}", msg);   // 这里可以自己使用第三方的logger库，写入到文件中  spd-logger
//			//abort();
//		}
//		else if (severity == Severity::kINFO) {
//			spdlog::info("Info: {}", msg);
//		}
//		else if (severity == Severity::kINTERNAL_ERROR) {
//			spdlog::error("Error: {}", msg);
//		}
//		else if (severity == Severity::kVERBOSE) {
//			spdlog::debug("Verbose: {}", msg);
//		}
//		else {
//			//Severity::kWARNING 
//			spdlog::warn("Warning: {}", msg);
//		}
//	}
//};

static Logger gLogger(Severity::kVERBOSE);

bool TRT::compile(const Mode& mode, unsigned int maxBatchSize, const std::string& onnx_file, const std::string& engine_save_path, const size_t maxWorkspaceSize)
{
	if (mode == Mode::INT8)
	{
		//INFOE("int8 not emplement.");
		spdlog::error("int8 not emplement.");
		return false;
	}

	spdlog::info("start compile...");

	//自定义删除函数
	std::shared_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(gLogger), destroy_nvidia_pointer<nvinfer1::IBuilder>);
	if (builder == nullptr) {
		spdlog::error("create builer error!");
		return false;
	}

	std::shared_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig(), destroy_nvidia_pointer<nvinfer1::IBuilderConfig>);

	if (mode == Mode::FP16) {
		if (builder->platformHasFastFp16())
		{
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}
		else
		{
			spdlog::warn("Plateform not support fp16");
		}
	}
	else if (mode == Mode::INT8) {
		if (builder->platformHasFastInt8())
		{
			config->setFlag(nvinfer1::BuilderFlag::kINT8);
		}
		else
		{
			spdlog::warn("Plateform not support int8");
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
		//spdlog::error("parse onnx file error!");
		return false;
	}

	// 打印一些输入 输出信息
	auto inputTensor = network->getInput(0);
	auto inputDims = inputTensor->getDimensions();

	spdlog::info("Input shape is {}", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());

	int num_input = network->getNbInputs();
	spdlog::info("Network has {} inputs", num_input);
	for (int i = 0; i < num_input; ++i) {
		auto tensor = network->getInput(i);
		auto dims = tensor->getDimensions();
		auto dim_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
		spdlog::info("input[{}] {} shape is {}", i, tensor->getName(), dim_str);
	}

	int num_output = network->getNbOutputs();
	spdlog::info("Network has {} outputs", num_output);
	for (int i = 0; i < num_output; ++i)
	{
		auto tensor = network->getOutput(i);
		auto dims = tensor->getDimensions();
		auto dim_str = join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
		spdlog::info("input[{}] {} shape is {}", i, tensor->getName(), dim_str);
	}

	// 打印每一层的维度
	int num_net_layers = network->getNbLayers();
	spdlog::info("Network has {} layers", num_net_layers);
	for (int i = 0; i < num_net_layers; ++i)
	{
		auto layer = network->getLayer(i);
		auto name = layer->getName();
		//auto input0 = layer->getInput(0);
		auto input = layer->getNbInputs();
		std::string input_dims = "";
		for (int j = 0; j < input; ++j)
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

		spdlog::info("[{}]  {} --> {}", name, input_dims, out_dims);
	}

	builder->setMaxBatchSize(maxBatchSize);
	spdlog::info("set MaxBatchSize = {}", maxBatchSize);

	config->setMaxWorkspaceSize(maxWorkspaceSize);
	spdlog::info("set MaxBatchSize = {}MB", maxBatchSize / 1024.0f / 1024.0f);

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

	spdlog::info("Build Engine.....");
	std::shared_ptr<nvinfer1::ICudaEngine> engine(builder->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<nvinfer1::ICudaEngine>);
	if (engine == nullptr)
	{
		spdlog::error("engine is nullptr");
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
		mode_name.replace(mode_name.find_last_of("."), 5, ".engine");
	}
	else
	{
		mode_name = "engine.engine";
	}
	std::string engine_file = engine_save_path + "/" + mode_str(mode) + "_" + mode_name;
	spdlog::info("Save engine to {}", engine_file);
	return save_file(engine_file, seridata->data(), seridata->size());
}
