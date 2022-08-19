#ifndef JOB_MANAGER_
#define JOB_MANAGER_
/*
* infer时，对job的管理。
* 实现为纯虚类：特定子类实现特定的方法 预处理和后处理等
* 共用函数：commit提交图像函数， fetch job等
* 需要一个队列存放处理后的数据。
* 多线程之间数据的同步等
* 模板
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
		std::shared_ptr<TRT::Tensor> tensor;  //  预处理之后的显存数据
		std::shared_ptr<std::promise<Output>> pro;
	};

	virtual ~JobManager(){
		//执行析构，join 所有线程
		stop();
	}

	void stop(){
		run_ = false;
		cond_.notify_all(); // 通知所有wait

		//  注意这里的 “区间”使用
		{
			// 这里使用unique_lock的原因：
			// 1. wait要求参数必须是unique_lock
			// 2. wait的内部实际上是反复转移mutex的归属权
			//     lock_grad引用提供的mutex，unique_lock指针指向mutex
			//     lock_grad不允许被copy移动，就是个简单的构造锁、析构解锁
			//     而unique_lock实现了移动语义转移所有权
		    // 	而wait函数内部需要转移mutex所有权的操作
			std::unique_lock<std::mutex>l(jobs_lock_);
			while (!jobs_.empty())
			{
				// 队列不为空，就得释放
				auto& item = jobs_.front();
				if (item.pro)
				{
					//当前job的future 还没赋值，给一个默认值
					item.pro->set_value(Output());
				}
				jobs_.pop();
			}
		}

		// 再停止子线程
		if (worker_)
		{
			worker_->join();
			worker_.reset();
		}
	}

	// 
	bool startup() {
		run_ = true; // 开始启动job管理线程
		std::promise<bool> pro;
		// 子线程启动 类成员函数（函数地址。this， 参数）
		worker_ = std::make_shared<std::thread>(&JobManager::worker,this, std::ref(pro));
		return pro.get_future().get(); // 取future的值。   没有有效值时阻塞到这里
		// 即如果worker函数 没有执行完成时，该函数就不会返回。
	}

	// std::shared_future 允许拷贝
	virtual std::shared_future<Output> commit(const Input& input) {
		// 提交图片，预处理后，放入队列
		Job job;
		job.pro = std::make_shared<std::promise<Output>>();
		if (!preprocess(job, input))
		{
			// 预处理完成后，才有后续操作
			// 否则处理失败。没有必要往队列中塞值了
			job.pro->set_value(Output());
			return job.pro->get_future();
		}
		
		{
			std::unique_lock<std::mutex> l(jobs_lock_);
			jobs_.push(job);
		}

		cond_.notify_one();  // 通知wait线程，队列有值了，可以继续了
		return job.pro->get_future();
	}

protected:
	// 不同子类实现自己的函数，返回的是是否启动成功
	virtual void worker(std::promise<bool>& result) = 0;
	virtual bool preprocess(Job& job, const Input& input) = 0;  // 对iunput预处理后，塞到job中

	// 从队列中取job，送往engine里面forward
	virtual bool get_jobs_and_wait(std::vector<Job>& fetch_jops, int max_batch) {
		// 从队列中取值，也需要加锁。
		std::unique_lock<std::mutex> l(jobs_lock_);

		// wait收到其他线程的 notify后，还需要保证其参数为true， 才会解除阻塞
		// 即： 队列不为空或者 run为false（子线程没有启动. 或者没有线程在启动。）。
		/*
		! 条件变量 wait函数
		! 等待notify信号的到来
		! 如果 notify信号到来，如果没来，就一直等
		! 流程是： 1. 判断条件，决定是否进入等待(看判断条件是否返回true)，如果不需要进入等待，返回true，wait继续持有锁，往下运行。返回false时，解锁，并阻塞
				   2. 解锁
		!          3. 等待信号到来,
		!             notify_one()则触发一个线程信号（顺序触发）。
		!             notify_all()触发所有等待的线程
		!          4. 根据情况加锁，并返回
		! 这里的条件就是wait函数 第二个参数的返回值
		! 进入等待的条件是：第二个参数的返回值为false时。
		! cv_.notify_one()  cv_.notify_all() 触发信号
		! 退出等待条是： 第二个参数的返回值为true

		wait()会去检查这些条件（lambda函数），当条件满足时（lambda返回true时），wait函数返回（继续往下走）。
		当条件不满足时，wait函数将解锁互斥量，并且将这个线程置于阻塞或者等待状态，当准备数据的线程调用notify通知条件变量时，处理数据的线程
		从睡眠中苏醒，重新获取互斥锁，并且再次检查lambda条件是否满足。在条件满足的情况下，wait返回，并继续持有锁
		当条件不满足时，线程将互斥量解锁，并重新开始等待。
		*/
		cond_.wait(l, [&]() {
			return !jobs_.empty() || !run_;
			});

		if (!run_)
		{
			// 没有线程处于启动状态
			return false;
		}

		// 到这里时，队列有值了，也有worker线程启动了 
		fetch_jops.clear();
		for (int i = 0; i < max_batch; ++i)
		{
			fetch_jops.emplace_back(std::move(jobs_.front()));  // 这里move了所有权。原来的第一个job就指向空了。
			jobs_.pop(); // pop 会调用析构，
		}
		return true;
	}

	virtual bool get_job_and_wait(Job& fetch_jobs) {
		// 从队列中取值，也需要加锁。
		std::unique_lock<std::mutex> l(jobs_lock_);
		cond_.wait(l, [&]() {
			return !jobs_.empty() || !run_;
			});

		if (!run_)
		{
			// 没有线程处于启动状态
			return false;
		}
		fetch_jobs = std::move(jobs_.front());
		jobs_.pop();
		return true;
	}
protected:
	InitParam init_param_;
	std::queue<Job> jobs_;  // 存放job队列
	
	// 多线程操作，【job管理线程和执行forward线程】
	std::shared_ptr<std::thread> worker_;
	// 多线程操作，所以用到锁
	std::mutex jobs_lock_;
	std::condition_variable cond_; // 条件变量，实现多线程之间数据交互
	std::atomic<bool> run_;   // 标志 子线程是否启动

};

#endif // !JOB_MANAGER_