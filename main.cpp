#include <iostream>
#include "trt_builder.hpp"
#include <queue>

class A
{
public:
	A() { std::cout << "aaaaaaaaa" << std::endl; };
	~A(){ std::cout << "bbbbbbbbbbbbb" << std::endl; };

	void p() { std::cout << "pppppppppp" << std::endl; }
private:

};



int main()
{


	std::queue<A> a;	
	A a1;
	a.push(a1);
	A a2;
	a.push(a2);
	A a3;
	a.push(a3);

	std::vector<A>  aaa;
	aaa.emplace_back(std::move(a.front()));
	a.pop();

	aaa[0].p();

	

	//TRT::compile(
	//	TRT::Mode::FP16, 
	//	1, 
	//	R"(F:\LearningCode\tensorRT_Pro\workspace\yolov5s.onnx)",
	//	R"(F:\LearningCode\tensorRT_Pro\workspace)"
	//);

	// ��ʼ��дinfer
	// ���÷�װ��˼��
		// ����engine�ļ�������engine
		// job����
		// tensor����

	system("pause");
	return 0;
}