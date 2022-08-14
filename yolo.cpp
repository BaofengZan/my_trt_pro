#include "yolo.hpp"
#include "yolov5_preprocess.h"
#include "job_management.hpp"
#include "cuda_tools.hpp"
#include <queue>
#include <memory>
#include <opencv2/opecv.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>



namespace Yolo {

    struct AffineMatrix {

    };


    using JobManagerImpl = JobManager<cv::Mat, BoxArray, std::tuple<std::string, int>, AffineMatrix>;
    class InferImpl : public Infer, public JobManagerImpl {
    public:
        // �ĸ����麯�� ����Ҫʵ��
        virtual void worker(std::promise<bool>& result) {



        }
        
        virtual bool preprocess(Job& job, const cv::Mat& input) {
            cudaSetDevice(gpu_id_);

            auto& tensor = job.tensor; // �����tensor ��δ����ռ�
            if (tensor == nullptr)
            {
                tensor = std::shared_ptr<TRT::Tensor>(new TRT::Tensor());
            }
            else
            {
                //�����Դ渴�ã��ͻᵽ������ڶ��׶�ʵ�֣�
            }

            // ��ʼԤ����
            //�õ�Ԥ������С֮�󣬷����ִ�
            tensor->resize({ 1,3,input_h_, input_w_ });

            // Ԥ��������
            // ����ԭʼͼ������
            // ����Ԥ����Ľ�����뵽tensor��
            uint8_t* img_device = nullptr;
            size_t  size_image = input.cols * input.rows * 3;
            checkCudaRuntime(cudaMalloc((void**)&img_device, size_image));
            cudaMemcpy(img_device, input.data, size_image, cudaMemcpyHostToDevice);
            //preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, stream);       
            preprocess_kernel_img(img_device, input.cols, input.rows, (float*)tensor->gpu(), input_w_, input_h_);
          

            cudaFree(img_device);
        } // ��iunputԤ���������job��
        
        virtual std::shared_future<BoxArray> commit(const cv::Mat& image) {}
        
        
        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) {}

    public:
        virtual bool startup(
            const std::string& engine_file,
            int gpuid,
            float confidence_threshold,
            float nms_threshold,
            int max_objects,
            bool use_multi_preprocess_stream
        ) {
            engine_file_ = engine_file;
            gpu_id_ = gpuid;
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            confidence_threshold_ = confidence_threshold;
            nms_threshold_ = nms_threshold;
            max_objects_ = max_objects;
            return JobManagerImpl::startup();
        }

    private:


    private:
        std::string engine_file_{ "" };
        int input_w_{ 0 };
        int input_h_{ 0 };
        int gpu_id_{ 0 };
        int confidence_threshold_{ 0 };
        int nms_threshold_{ 0 };
        int  max_objects_{ 1024 };
        bool use_multi_preprocess_stream_{ false };

    };
    
    
    
    std::shared_ptr<Infer> Yolo::create_infer(const std::string& engine_file, int gpuid, float condifence_threshold, float nms_threshold, int max_objects, bool use_multi_preprocess_stream)
    {
        std::shared_ptr<InferImpl> instace(new InferImpl());
        if (!instace->startup(engine_file, gpuid, condifence_threshold, nms_threshold, max_objects, use_multi_preprocess_stream))
        {
            instace.reset();  // ����ʧ�ܣ�����
        }
        return instace;
    }

};

