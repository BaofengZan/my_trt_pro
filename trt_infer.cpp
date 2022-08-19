#include "trt_infer.hpp"
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
#include <assert.h>

// trt �涨��Ҫ�̳�ILogger��ʵ��logger
// virtual void log(Severity severity, AsciiChar const* msg) noexcept = 0;
// AsciiChar const* == const char*
class Logger : public nvinfer1::ILogger {

public:
	virtual void log(Severity severity, const char* msg) noexcept {
		if (severity == Severity::kERROR) {
			printf("Error: %s\n", msg);   // ��������Լ�ʹ�õ�������logger�⣬д�뵽�ļ���  spd-logger
			abort();
		}
		else if (severity == Severity::kINFO) {
			printf("Info: %s\n", msg);
		}
		else if (severity == Severity::kINTERNAL_ERROR) {
			printf("Error: %s\n", msg);
		}
		else if (severity == Severity::kVERBOSE) {
			printf("Verbose: %s\n", msg);
		}
		else {
			//Severity::kWARNING 
			printf("Warning: %s\n", msg);
		}
	}
};

static Logger gLogger;


namespace TRT {

	template<typename T>
	static void destroy_nvidia_pointer(T* ptr) {
		if (ptr)
		{
			ptr->destroy();
		}
	}


	template<typename T>
	static void destroy_nvidia_pointer_trt8(T* ptr) {
		if (ptr)
		{
			delete ptr;
		}
	}

	class InferImpl : public Infer {
	
	public:
		~InferImpl()
		{
			destroy();
		}
		virtual void forward() override;
		virtual int get_max_batch_size() override;
		virtual int get_input_h() override;
		virtual int get_input_w() override;
		virtual std::shared_ptr<TRT::Tensor> tensor(const std::string& name) override;

		bool load(const std::string& engine_file);
	private:
		void destroy();
		bool build_engie(const void* pdata, size_t size);

		void build_engine_input_and_outputs_mapper();
	private:

		std::vector<std::shared_ptr<TRT::Tensor>> inputs_;
		std::vector<std::shared_ptr<TRT::Tensor>> outputs_;
		std::vector<std::string> inputs_name_;
		std::vector<std::string> outputs_name_;
 
		std::map<std::string, int> layername_index_mapper_;  // ֻ��Ҫ���������
		std::vector<std::shared_ptr<TRT::Tensor>> order_layers_;
	

		std::vector<void*> bindingsPtr_;

		std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
		std::shared_ptr<nvinfer1::IExecutionContext> context_{nullptr};
		std::shared_ptr<nvinfer1::IRuntime> runtime_{nullptr};

	};




    std::shared_ptr<Infer> TRT::create_engine(const std::string& engine_file)
    {
		std::shared_ptr<InferImpl> Infer(new InferImpl());
		if (!Infer->load(engine_file))
		{
			Infer.reset();
				
		}
        return Infer;
    }


	void InferImpl::forward()
	{
		for (int i = 0; i < order_layers_.size(); ++i)
		{
			bindingsPtr_[i] = order_layers_[i]->gpu();
		}
		void** bindingsptr = bindingsPtr_.data();
		bool execute_result = context_->enqueueV2(bindingsptr, nullptr, nullptr);
		if (!execute_result) {
			auto code = cudaGetLastError();
			printf("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
		}
	}

	int InferImpl::get_max_batch_size()
	{
		assert(context_ != nullptr);
		return engine_->getMaxBatchSize();
	}

	int InferImpl::get_input_h()
	{
		assert(context_ != nullptr);
		return engine_->getBindingDimensions(0).d[2];
	}

	int InferImpl::get_input_w()
	{
		assert(context_ != nullptr);
		return engine_->getBindingDimensions(0).d[3];
	}

	std::shared_ptr<TRT::Tensor> InferImpl::tensor(const std::string& name)
	{
		auto node = layername_index_mapper_.find(name);
		if (node == layername_index_mapper_.end())
		{
			printf("Could not found the input/output node '%s', please makesure your model", name.c_str());
		}
		return order_layers_[node->second];
	}

	bool InferImpl::load(const std::string& engine_file)
	{
		// �����ļ�
		auto data = load_file(engine_file);

		auto ret = build_engie(data.data(), data.size());
		if (!ret)
		{
			context_.reset();
			return false;
		}
		// ����layer name �� index��ӳ��
		build_engine_input_and_outputs_mapper();
		return true;
	}

	void InferImpl::destroy()
	{
		inputs_.clear();
		outputs_.clear();
		inputs_name_.clear();
		outputs_name_.clear();

		layername_index_mapper_.clear();
		order_layers_.clear();

		bindingsPtr_.clear();

		engine_.reset();
		context_.reset();
		runtime_.reset();
	}

	bool InferImpl::build_engie(const void* pdata, size_t size)
	{
		if (pdata == nullptr || size == 0) {
			printf("data is null\n");
			return false;
		}

		runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger), destroy_nvidia_pointer_trt8<nvinfer1::IRuntime>);
		if (runtime_ == nullptr)
		{
			return false;
		}

		engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size), destroy_nvidia_pointer_trt8<nvinfer1::ICudaEngine>);
		if (engine_ == nullptr)
		{
			return false;
		}

		context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(), destroy_nvidia_pointer_trt8<nvinfer1::IExecutionContext>);
		if (context_ == nullptr)
		{
			return false;
		}
		return true;
	}

	void InferImpl::build_engine_input_and_outputs_mapper()
	{
		int nbBindings = engine_->getNbBindings();  // �����������
		int max_batch_size = engine_->getMaxBatchSize();

		inputs_.clear();
		outputs_.clear();
		inputs_name_.clear();
		outputs_name_.clear();
		layername_index_mapper_.clear();
		order_layers_.clear();
		bindingsPtr_.clear();

		for (int i = 0; i < nbBindings; ++i)
		{
			auto dims = engine_->getBindingDimensions(i); // ȡά����Ϣ
			auto type = engine_->getBindingDataType(i);  // ��������Ĭ��ȫ����flaot

			auto binding_name = engine_->getBindingName(i);
			
			dims.d[0] = max_batch_size;
			auto new_tensor = std::make_shared<TRT::Tensor>(dims);
			if (engine_->bindingIsInput(i))
			{
				inputs_.push_back(new_tensor);
				inputs_name_.push_back(binding_name);
			}
			else
			{
				outputs_.push_back(new_tensor);
				outputs_name_.push_back(binding_name);
			}
			
			layername_index_mapper_[binding_name] = i;
			order_layers_.push_back(new_tensor);
		}
		bindingsPtr_.resize(order_layers_.size());
	}

};



