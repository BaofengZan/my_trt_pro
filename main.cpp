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

	// 开始编写infer
	// 利用封装的思想
		// 解析engine文件，生成engine
		// job管理。
		// tensor管理

	system("pause");
	return 0;
}