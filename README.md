# 跟着大佬学知识

* 一步一步实现Tensor_pro，从而理解手写AI团队的思想
    * 我仅上传代码：在windows下，用vs创建工程然后加载代码；在linux下，根据自己习惯创建工程。

* 深入理解线程、内存复用等知识
    * 无锁队列

* 第一阶段记录  [V1](./docs/v1.md) 
    * 基本流程，job管理 

* 第二阶段记录 [V2](./docs/v2.md)
    * tensor的管理

* 第三阶段记录 [V3](./docs/v3.md)
    * 显存的复用

* nvidia解码

## bug 记录

```c
/*
trt_infet.cpp
因为我们的onnx是explicite 模式。
所以在enqueueV2之前必须要setBindingDimensions
*/
int inputBatchSize = inputs_[0]->size(0);  // 拿到batch
for (int i = 0; i < engine_->getNbBindings(); ++i) {
    auto dims = engine_->getBindingDimensions(i);
    auto type = engine_->getBindingDataType(i);
    dims.d[0] = inputBatchSize;
    if (engine_->bindingIsInput(i)) {
        context_->setBindingDimensions(i, dims);
    }
}
```

将输入的shape重置后，那么output的大小也要改变，此时对Tensor的管理中，就得支持resize_dim的操作

# 引用

[shouxieai/tensorRT_Pro: C++ library based on tensorrt integration](https://github.com/shouxieai/tensorRT_Pro)

感谢手写AI各位大佬的无私贡献，让我们可以学习到更深入的知识！！！
