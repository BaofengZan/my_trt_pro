#include "cuda_tools.hpp"
#include "yolo.hpp"
#include "yolov5_preprocess.h"
#include "job_management.hpp"
#include "trt_infer.hpp"
#include <queue>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "log.h"


namespace Yolo {

    struct AffineMatrix {

    };

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold,
        float* invert_affine_matrix, float* parray,
        int max_objects, cudaStream_t stream
    );

    static float iou(const Box& a, const Box& b) {
        float cleft = std::max(a.left, b.left);
        float ctop = std::max(a.top, b.top);
        float cright = std::min(a.right, b.right);
        float cbottom = std::min(a.bottom, b.bottom);

        float c_area = std::max(cright - cleft, 0.0f) * std::max(cbottom - ctop, 0.0f);
        if (c_area == 0.0f)
            return 0.0f;

        float a_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top);
        float b_area = std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top);
        return c_area / (a_area + b_area - c_area);
    }

    static BoxArray cpu_nms(BoxArray& boxes, float threshold) {

        std::sort(boxes.begin(), boxes.end(), [](BoxArray::const_reference a, BoxArray::const_reference b) {
            return a.confidence > b.confidence;
            });

        BoxArray output;
        output.reserve(boxes.size());

        std::vector<bool> remove_flags(boxes.size());
        for (int i = 0; i < boxes.size(); ++i) {

            if (remove_flags[i]) continue;

            auto& a = boxes[i];
            output.emplace_back(a);

            for (int j = i + 1; j < boxes.size(); ++j) {
                if (remove_flags[j]) continue;

                auto& b = boxes[j];
                if (b.class_label == a.class_label) {
                    if (iou(a, b) >= threshold)
                        remove_flags[j] = true;
                }
            }
        }
        return output;
    }


    using JobManagerImpl = JobManager<cv::Mat, BoxArray, std::tuple<std::string, int>, AffineMatrix>;
    class InferImpl : public Infer, public JobManagerImpl {
    public:
        virtual ~InferImpl()
        {
            // 要求在inferImpl 里面执行stop，而不是在基类执行
            stop();
        }
        // 四个纯虚函数 必须要实现
        virtual void worker(std::promise<bool>& result) {
            checkCudaRuntime(cudaSetDevice(gpu_id_));

            // 后面实现
            auto engine = TRT::create_engine(engine_file_); // 解析文件，生成engien
            if (engine == nullptr) {
                spdlog::error("Engine {} load failed", engine_file_);
                result.set_value(false);
                return;
            }
            auto input = engine->tensor("images");
            auto output = engine->tensor("output");
            int num_classes = 80;

            //engine->print();  // 打印engine的一些信息

            const int MAX_IMAGE_BOXES = max_objects_;
            const int NUM_BOX_ELEMENT = 7; //  // left, top, right, bottom, confidence, class, keepflag  
            // keepfalg 用于nms中的

            // 申请一大块显存
            int max_batch_size = engine->get_max_batch_size();
            TRT::Tensor output_array_device;
            input_h_ = engine->get_input_h();
            input_w_ = engine->get_input_w();
            tensor_allocator_ = std::make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);

            result.set_value(true);


            // 输出的维度是，batch*(1+NUM_BOX_ELEMENT*MAX_IMAGE_BOXES)  都是float
            // 目前先实现batch=1
            output_array_device.resize({ max_batch_size, 1 + NUM_BOX_ELEMENT * MAX_IMAGE_BOXES });

            std::vector<Job> fetch_jobs;
            //while (get_jobs_and_wait(fetch_jobs, max_batch_size))
            Job fetch_job;
            while (get_job_and_wait(fetch_job))
            {
                //for (int ibatch = 0; ibatch < fetch_jobs.size(); ++ibatch)
                //{
                //    auto& job = fetch_jobs[ibatch];
                //    auto& mono = job.tensor;
                //}
                //auto& mono = fetch_job.tensor;
                auto& mono = fetch_job.tensor->data();
                spdlog::info("mono dim = {} {} {} {}", mono->size(0), mono->size(1), mono->size(2), mono->size(3));
                // 拿到了tensor
                //cudaMemcpy(input->gpu(), mono->gpu(), mono->byte_size(), cudaMemcpyDeviceToDevice);
                input->set_data(mono->gpu(), mono->byte_size(), TRT::DataTransType::D2D);

                //这块显存用完就释放
                fetch_job.tensor->release();
                
                //std::vector<float> cpu_out;
                //int size = input->byte_size() / sizeof(float);
                //cpu_out.resize(size);
                //cudaMemcpy(cpu_out.data(), input->gpu(), input->byte_size(), cudaMemcpyDeviceToHost);
                //for (int i = 0; i < cpu_out.size(); ++i)
                //{
                //    printf("val = %f\n", cpu_out[i]);
                //}


                engine->forward();

                float* image_based_output = (float*)output->gpu();



                decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, nullptr, (float*)output_array_device.gpu(), MAX_IMAGE_BOXES, nullptr);


              /*  std::vector<float> cpu_out;
                int size = output_array_device.byte_size() / sizeof(float);

                cpu_out.resize(size);

                cudaMemcpy(cpu_out.data(), output_array_device.gpu(), output_array_device.byte_size(), cudaMemcpyDeviceToHost);
                */

                float* cpu_out = output_array_device.cpu<float>();
                int size = output_array_device.numel(); // 个数
                //for (int i = 0; i < cpu_out.size(); ++i)
                //{
                //    printf("val = %f\n", cpu_out[i]);
                //}
                int count = std::min(MAX_IMAGE_BOXES, (int)cpu_out[0]);
                auto& image_based_boxes = fetch_job.output;
                for (int i = 0; i < count; ++i) {
                   float* pbox = output_array_device.cpu<float>(1 + i * NUM_BOX_ELEMENT);
                    int label = pbox[5];
                    int keepflag = pbox[6];
                    if (keepflag == 1) {
                        image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    }
                }

                image_based_boxes = cpu_nms(image_based_boxes, nms_threshold_);

                fetch_job.pro->set_value(image_based_boxes);
            }

            tensor_allocator_.reset();
        }

        virtual bool preprocess(Job& job, const cv::Mat& input) {
            cudaSetDevice(gpu_id_);

            // 先申请一块
            job.tensor = tensor_allocator_->query();
            if (job.tensor == nullptr) {
                spdlog::info("Tensor allocator query failed.");
                return false;
            }

            auto& tensor = job.tensor->data(); // 拿到tensor的指针
            if (tensor == nullptr)
            {
                tensor = std::shared_ptr<TRT::Tensor>(new TRT::Tensor(TRT::DataType::Float));
            }
            else
            {
                //若有显存复用，就会到这里，（第二阶段实现）
            }

            // 开始预处理
            //得到预处理大大小之后，分配现存
            tensor->resize({ 1,3,input_h_, input_w_ });

            // 预处理函数。
            // 处理原始图像数据
            // 最后把预处理的结果放入到tensor中
            //uint8_t* img_device = nullptr;
            //int  size_image = input.cols * input.rows * 3;  // uint8 sizeof=1
            //checkCudaRuntime(cudaMalloc((void**)&img_device, size_image));
            //cudaMemcpy(img_device, input.data, size_image, cudaMemcpyHostToDevice);
            // preprocess_kernel_img(img_device, input.cols, input.rows, (float*)tensor->gpu(), input_w_, input_h_, nullptr);
            //cudaFree(img_device);

            tensor->set_workspace<uint8_t>({1,3, input.rows, input.cols }, input.data);
            preprocess_kernel_img(tensor->get_workspace<uint8_t>(), input.cols, input.rows, (float*)tensor->gpu(), input_w_, input_h_, nullptr);
            return true;
        } // 对iunput预处理后，塞到job中

        virtual std::shared_future<BoxArray> commit(const cv::Mat& image) {
            return JobManager::commit(image);
        }


        virtual std::vector<std::shared_future<BoxArray>> commits(const std::vector<cv::Mat>& images) {
            // 暂时未实现
            return std::vector<std::shared_future<BoxArray>>();
        }

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
        float confidence_threshold_{ 0 };
        float nms_threshold_{ 0 };
        int  max_objects_{ 1024 };
        bool use_multi_preprocess_stream_{ false };

    };



    std::shared_ptr<Infer> Yolo::create_infer(const std::string& engine_file, int gpuid, float condifence_threshold, float nms_threshold, int max_objects, bool use_multi_preprocess_stream)
    {
        spdlog::info("create infer");
        std::shared_ptr<InferImpl> instace(new InferImpl());
        if (!instace->startup(engine_file, gpuid, condifence_threshold, nms_threshold, max_objects, use_multi_preprocess_stream))
        {
            instace.reset();  // 启动失败，重置
        }
        return instace;
    }

};

