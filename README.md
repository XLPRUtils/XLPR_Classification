# XLPR_Classification：Image Recognition Classification Framework - Basic Edition
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![cuda-image]][cuda-url]
[![torchvision-image]][torchvision-url]
[![tensorflow-image]][tensorflow-url]


**Atari has topped itself with SwordQuest!**  
在这个教程中中将演示通过该分类框架实现简易的手势识别。方便大家了解深度学习以及计算机视觉。

## 安装
```
# 在开始安装前，请确认操作系统为Ubuntu，且已经安装Anaconda

# 创建运行虚拟环境，建议python=3.7，名称可以自己调整
conda create -n classify python=3.7

# 激活环境
source activate classify

# 安装相关依赖环境（Pytorch、TF2.0、CUDA等）
安装教程请参考[环境安装比较](https://www.yuque.com/docs/share/41faffd5-271f-4853-9832-450a4f296f06?# 《配置PyTorch深度学习环境》)	

# 环境依赖：Pytorch>=1.6,TensorFlow>=2.0, CUDA(兼容), prettytable, sklearn, albumentations, cv2, matplotlib

# 如果报错，请调整上述环境依赖包的版本或联系神奇海螺
```

## 使用
-----------------
### 框架概述
- **主函数接口**  
```
train.py：进行模型的训练  *CUDA_VISIBLE_DEVICES=0 python train.py*
inference.py： 进行模型的预测与评估 *CUDA_VISIBLE_DEVICES=0 python inference.py*
mean_std.py: 计算数据集均值与方差，进行z-score 标准化 *python mean_std.py* 
详细的代码解释与可参考[框架教程]("https://www.yuque.com/docs/share/abb954bf-bdbc-4cc8-98e1-7f5ef2de081d?# 《ImageClassification（basis）》")
```

- **重要文件夹概述**  
```
.{ROOT}
├── Dataloaders
│   ├── Dataloader.py 数据集索引文件 
│   ├── GestureDataSet.py 手势识别Dataloader
├── Networks
│   ├── Model.py 模型索引文件 
│   ├── ResNet.py
│   ├── ...
│   ├── ...
│   └── DenseNet.py 
├── Utils
│   ├── Criterion.py 损失函数
│   ├── Optimizer.py 优化器
│   ├── Scheduler.py 学习策略
│   ├── TextLogger.py Log日志
│   ├── Trainer.py 训练验证过程
│   └── utils.py 初始化文件
├── test
│   ├── linux.jpg Linus Benedict Torvalds国际友好手势
│   ├── wakan.jpg Vulcan 举手礼星际友好手势
├── train.py 训练验证文件
├── inference.py 预测文件
├── mean_std.py 计算均值与标准差
```


<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
-->

[python-image]: https://img.shields.io/badge/Python-3.7.7-blue.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.6-red.svg
[pytorch-url]: https://pytorch.org/
[cuda-image]: https://img.shields.io/badge/CUDA-10.2-green.svg
[cuda-url]: https://developer.nvidia.com/
[torchvision-image]: https://img.shields.io/badge/torchvision-0.6-orange.svg
[torchvision-url]: https://pytorch.org/docs/stable/torchvision/index.html/
[tensorflow-image]: https://img.shields.io/badge/tensorflow-2.1-yellow.svg
[tensorflow-url]: https://tensorflow.google.cn/
