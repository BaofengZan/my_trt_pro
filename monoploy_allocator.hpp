#ifndef MONOPOLY_ALLOCATOR_HPP_
#define MONOPOLY_ALLOCATOR_HPP_

/*
* 显存复用，
* 因为有多个线程会来申请空间，所以需要有个计数器，（就涉及到了多线程写一个变量：需要锁/条件变量）
* 对每一块cell 我们需要有个可用不可用的标志
*/


#include <vector>
#include <memory>
#include <condition_variable>
#include <mutex>

template <typename _ItemType>
class MonopolyAllocator {
public:
	// MonopolyData  改为cell 方便理解
	class CellData {
		std::shared_ptr<_ItemType>& data() { return data_; }
		// 用完后，我们要将该cell的available 置为true
		// 释放所有权
		// 释放后，不仅仅设置这个，还要设置可用cell的数量，所以
		// 实际释放不应该在这里，而是在外面。 所以用到friend class
		void release() { manager_->release(); }
	private:
		CellData(MonopolyAllocator* manager) { manager_ = manager; };

		friend class MonopolyAllocator;
		MonopolyAllocator* manager_{ nullptr };
	private:
		std::shared_ptr<_ItemType> data_{nullptr};  // float* data
		bool available_{ true };                    // 是否可用
	};


	MonopolyAllocator(int size) {
		// 初始化，
		capacity_ = size;
		num_available_ = size;
		datas_.resize(size);

		for (int i = 0; i < size; ++i) {
			datas_[i] = std::shared_ptr<CellData>();
		}
	}

	~MonopolyAllocator(){
		run_ = false;
		cv_.notify_all();

		std::unique_lock<std::mutex> l(lock_);
		// 在num_wait_thread_为0时，持有锁，最后析构。
		cv_exit_.wait(l, [&]() {
			return num_wait_thread_ == 0;
			});
	}

	/*
	*
	* timeout：超时时间，如果没有可用的对象，将会进入阻塞等待，如果等待超时则返回空指针
	* wait_for
	*/
	std::shared_ptr<CellData> query(int timeout = 10000) {
		std::unique_lock<std::mutex> l(lock_);
		if (!run) { return nullptr; } // 此时任务没有run

		if (num_available_ == 0) {
			// 此时没有可用的cell
			num_available_++;
			auto state = cv_.wait_for(l, std::chrono::milliseconds(timeout), [&]()
				{
					return num_available_ > 0 || !run;
				});

			// 到这里时，说明有cell可用了
			num_available_--;
			cv_exit_.notify_one();

			if (!state || num_available_==0||!run)
			{
				return nullptr;
			}
		}

		auto item = std::find_if(datas_.begin(), datas_.end() [](std::shared_ptr<CellData>& item) {return item->available_; });
		if (item == datas_.end())
		{
			return nullptr;
		}

		(*item)->available = false;  // 被使用了
		num_available_--;
		return *item;


	}
	
	
	int num_avialable() { return num_available_; }
	int capacity() { return capacity_; };
private:
	void release(CellData* cell) {
		std::unique_lock<std::mutex> l(lock_);
		if (!cell->available_){
			cell->available_ = true;
			num_available_++;
			cv_.notify_one();
		}
	}
private:
	std::mutex lock_;
	std::condition_variable cv_;
	std::condition_variable cv_exit_;  // 退出时 使用
	std::vector<std::shared_ptr<CellData>> datas_;

	int capacity_{ 0 };

	/*
	* volatile
		- 保证变量的内存可见性 - 禁止指令重排序
	* 遇到这个关键字声明的变量，编译器对访问该变量的代码就不再进行优化，从而可以提供对特殊地址的稳定访问
	* 多任务环境下各任务间共享的标志应该加 volatile；
		- volatile 的意思是让编译器每次操作该变量时一定要从内存中真正取出，而不是使用已经存在寄存器中的值
	* 中断服务程序中修改的供其它程序检测的变量需要加 volatile；
	* 存储器映射的硬件寄存器通常也要加 volatile 说明，因为每次对它的读写都可能由不同意义；
	*/ 
	volatile int num_available_{ 0 };  // 可用的cell数量
	volatile int num_wait_thread_{ 0 };
	volatile bool run_{ true };
};



#endif // !MONOPOLY_ALLOCATOR_HPP_
