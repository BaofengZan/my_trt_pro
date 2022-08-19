#ifndef JOB_MANAGER_
#define JOB_MANAGER_
/*
* inferʱ����job�Ĺ���
* ʵ��Ϊ�����ࣺ�ض�����ʵ���ض��ķ��� Ԥ����ͺ����
* ���ú�����commit�ύͼ������ fetch job��
* ��Ҫһ�����д�Ŵ��������ݡ�
* ���߳�֮�����ݵ�ͬ����
* ģ��
*/

#include <string>
#include <vector>
#include <memory>
#include <queue>
#include <thread>
#include <future>
#include <condition_variable>
#include "trt_tensor.hpp"

template <class Input, class Output, class InitParam = std::tuple<std::string, int>, class WarpAffineMatrix=int >
class JobManager {
public:
	struct Job
	{
		Input input;  //Mat
		Output output;   //Box
		std::shared_ptr<TRT::Tensor> tensor;  //  Ԥ����֮����Դ�����
		std::shared_ptr<std::promise<Output>> pro;
	};

	virtual ~JobManager(){
		//ִ��������join �����߳�
		stop();
	}

	void stop(){
		run_ = false;
		cond_.notify_all(); // ֪ͨ����wait

		//  ע������� �����䡱ʹ��
		{
			// ����ʹ��unique_lock��ԭ��
			// 1. waitҪ�����������unique_lock
			// 2. wait���ڲ�ʵ�����Ƿ���ת��mutex�Ĺ���Ȩ
			//     lock_grad�����ṩ��mutex��unique_lockָ��ָ��mutex
			//     lock_grad������copy�ƶ������Ǹ��򵥵Ĺ���������������
			//     ��unique_lockʵ�����ƶ�����ת������Ȩ
		    // 	��wait�����ڲ���Ҫת��mutex����Ȩ�Ĳ���
			std::unique_lock<std::mutex>l(jobs_lock_);
			while (!jobs_.empty())
			{
				// ���в�Ϊ�գ��͵��ͷ�
				auto& item = jobs_.front();
				if (item.pro)
				{
					//��ǰjob��future ��û��ֵ����һ��Ĭ��ֵ
					item.pro->set_value(Output());
				}
				jobs_.pop();
			}
		}

		// ��ֹͣ���߳�
		if (worker_)
		{
			worker_->join();
			worker_.reset();
		}
	}

	// 
	bool startup() {
		run_ = true; // ��ʼ����job�����߳�
		std::promise<bool> pro;
		// ���߳����� ���Ա������������ַ��this�� ������
		worker_ = std::make_shared<std::thread>(&JobManager::worker,this, std::ref(pro));
		return pro.get_future().get(); // ȡfuture��ֵ��   û����Чֵʱ����������
		// �����worker���� û��ִ�����ʱ���ú����Ͳ��᷵�ء�
	}

	// std::shared_future ������
	virtual std::shared_future<Output> commit(const Input& input) {
		// �ύͼƬ��Ԥ����󣬷������
		Job job;
		job.pro = std::make_shared<std::promise<Output>>();
		if (!preprocess(job, input))
		{
			// Ԥ������ɺ󣬲��к�������
			// ������ʧ�ܡ�û�б�Ҫ����������ֵ��
			job.pro->set_value(Output());
			return job.pro->get_future();
		}
		
		{
			std::unique_lock<std::mutex> l(jobs_lock_);
			jobs_.push(job);
		}

		cond_.notify_one();  // ֪ͨwait�̣߳�������ֵ�ˣ����Լ�����
		return job.pro->get_future();
	}

protected:
	// ��ͬ����ʵ���Լ��ĺ��������ص����Ƿ������ɹ�
	virtual void worker(std::promise<bool>& result) = 0;
	virtual bool preprocess(Job& job, const Input& input) = 0;  // ��iunputԤ���������job��

	// �Ӷ�����ȡjob������engine����forward
	virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jops, int max_batch) {
		// �Ӷ�����ȡֵ��Ҳ��Ҫ������
		std::unique_lock<std::mutex> l(jobs_lock_);

		// wait�յ������̵߳� notify�󣬻���Ҫ��֤�����Ϊtrue�� �Ż�������
		// ���� ���в�Ϊ�ջ��� runΪfalse�����߳�û������. ����û���߳�������������
		/*
		! �������� wait����
		! �ȴ�notify�źŵĵ���
		! ��� notify�źŵ��������û������һֱ��
		! �����ǣ� 1. �ж������������Ƿ����ȴ�(���ж������Ƿ񷵻�true)���������Ҫ����ȴ�������true��wait�������������������С�����falseʱ��������������
				   2. ����
		!          3. �ȴ��źŵ���,
		!             notify_one()�򴥷�һ���߳��źţ�˳�򴥷�����
		!             notify_all()�������еȴ����߳�
		!          4. �������������������
		! �������������wait���� �ڶ��������ķ���ֵ
		! ����ȴ��������ǣ��ڶ��������ķ���ֵΪfalseʱ��
		! cv_.notify_one()  cv_.notify_all() �����ź�
		! �˳��ȴ����ǣ� �ڶ��������ķ���ֵΪtrue

		wait()��ȥ�����Щ������lambda������������������ʱ��lambda����trueʱ����wait�������أ����������ߣ���
		������������ʱ��wait���������������������ҽ�����߳������������ߵȴ�״̬����׼�����ݵ��̵߳���notify֪ͨ��������ʱ���������ݵ��߳�
		��˯�������ѣ����»�ȡ�������������ٴμ��lambda�����Ƿ����㡣���������������£�wait���أ�������������
		������������ʱ���߳̽������������������¿�ʼ�ȴ���
		*/
		cond_.wait(l, [&]() {
			return !jobs_.empty() || !run_;
			});

		if (!run_)
		{
			// û���̴߳�������״̬
			return false;
		}

		// ������ʱ��������ֵ�ˣ�Ҳ��worker�߳������� 
		fetch_jops.clear();
		for (int i = 0; i < max_batch; ++i)
		{
			fetch_jops.emplace_back(std::move(jobs_.front()));  // ����move������Ȩ��ԭ���ĵ�һ��job��ָ����ˡ�
			jobs_.pop(); // pop �����������
		}
		return true;
	}

	virtual bool get_job_and_wait(Job& fetch_jobs) {
		// �Ӷ�����ȡֵ��Ҳ��Ҫ������
		std::unique_lock<std::mutex> l(jobs_lock_);
		cond_.wait(l, [&]() {
			return !jobs_.empty() || !run_;
			});

		if (!run_)
		{
			// û���̴߳�������״̬
			return false;
		}
		fetch_jobs = std::move(jobs_.front());
		jobs_.pop();
		return true;
	}
protected:
	InitParam init_param_;
	std::queue<Job> jobs_;  // ���job����
	
	// ���̲߳�������job�����̺߳�ִ��forward�̡߳�
	std::shared_ptr<std::thread> worker_;
	// ���̲߳����������õ���
	std::mutex jobs_lock_;
	std::condition_variable cond_; // ����������ʵ�ֶ��߳�֮�����ݽ���
	std::atomic<bool> run_;   // ��־ ���߳��Ƿ�����

};

#endif // !JOB_MANAGER_