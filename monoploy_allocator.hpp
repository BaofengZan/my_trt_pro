#ifndef MONOPOLY_ALLOCATOR_HPP_
#define MONOPOLY_ALLOCATOR_HPP_

/*
* �Դ渴�ã�
* ��Ϊ�ж���̻߳�������ռ䣬������Ҫ�и��������������漰���˶��߳�дһ����������Ҫ��/����������
* ��ÿһ��cell ������Ҫ�и����ò����õı�־
*/


#include <vector>
#include <memory>
#include <condition_variable>
#include <mutex>

template <typename _ItemType>
class MonopolyAllocator {
public:
	// MonopolyData  ��Ϊcell �������
	class CellData {
		std::shared_ptr<_ItemType>& data() { return data_; }
		// ���������Ҫ����cell��available ��Ϊtrue
		// �ͷ�����Ȩ
		// �ͷź󣬲����������������Ҫ���ÿ���cell������������
		// ʵ���ͷŲ�Ӧ����������������档 �����õ�friend class
		void release() { manager_->release(); }
	private:
		CellData(MonopolyAllocator* manager) { manager_ = manager; };

		friend class MonopolyAllocator;
		MonopolyAllocator* manager_{ nullptr };
	private:
		std::shared_ptr<_ItemType> data_{nullptr};  // float* data
		bool available_{ true };                    // �Ƿ����
	};


	MonopolyAllocator(int size) {
		// ��ʼ����
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
		// ��num_wait_thread_Ϊ0ʱ�������������������
		cv_exit_.wait(l, [&]() {
			return num_wait_thread_ == 0;
			});
	}

	/*
	*
	* timeout����ʱʱ�䣬���û�п��õĶ��󣬽�����������ȴ�������ȴ���ʱ�򷵻ؿ�ָ��
	* wait_for
	*/
	std::shared_ptr<CellData> query(int timeout = 10000) {
		std::unique_lock<std::mutex> l(lock_);
		if (!run) { return nullptr; } // ��ʱ����û��run

		if (num_available_ == 0) {
			// ��ʱû�п��õ�cell
			num_available_++;
			auto state = cv_.wait_for(l, std::chrono::milliseconds(timeout), [&]()
				{
					return num_available_ > 0 || !run;
				});

			// ������ʱ��˵����cell������
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

		(*item)->available = false;  // ��ʹ����
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
	std::condition_variable cv_exit_;  // �˳�ʱ ʹ��
	std::vector<std::shared_ptr<CellData>> datas_;

	int capacity_{ 0 };

	/*
	* volatile
		- ��֤�������ڴ�ɼ��� - ��ָֹ��������
	* ��������ؼ��������ı������������Է��ʸñ����Ĵ���Ͳ��ٽ����Ż����Ӷ������ṩ�������ַ���ȶ�����
	* �����񻷾��¸�����乲��ı�־Ӧ�ü� volatile��
		- volatile ����˼���ñ�����ÿ�β����ñ���ʱһ��Ҫ���ڴ�������ȡ����������ʹ���Ѿ����ڼĴ����е�ֵ
	* �жϷ���������޸ĵĹ�����������ı�����Ҫ�� volatile��
	* �洢��ӳ���Ӳ���Ĵ���ͨ��ҲҪ�� volatile ˵������Ϊÿ�ζ����Ķ�д�������ɲ�ͬ���壻
	*/ 
	volatile int num_available_{ 0 };  // ���õ�cell����
	volatile int num_wait_thread_{ 0 };
	volatile bool run_{ true };
};



#endif // !MONOPOLY_ALLOCATOR_HPP_
