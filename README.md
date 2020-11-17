# Swordquest：Semantic segmentation integration framework by pytorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![cuda-image]][cuda-url]
[![torchvision-image]][torchvision-url]
[![tensorflow-image]][tensorflow-url]


**Atari has topped itself with SwordQuest!**  
This project aims at providing a concise, easy-to-use, modifiable reference implementation for semantic segmentation models using PyTorch.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## 安装
```
# 在开始安装前，请确认操作系统为Ubuntu，且已经安装Anaconda

# 创建运行虚拟环境，建议python=3.7.7，名称可以自己调整
conda create -n segmenter python=3.7.7

# 激活环境
source activate segmenter

# 安装依赖包，这里只是局部，如果缺少什么请自行conda或pip安装
pip install ninja tqdm

# 安装Pytorch和torchvision 具体安装可以参考https://pytorch.org/
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch

# 该项目有部分内容需要编译，编译Semantic segmentation 0.1包
cd awesome-semantic-segmentation-pytorch/core/nn
python setup.py build develop

# 如果报错，请调整上述环境依赖包的版本或联系神奇海螺
```

## 使用
-----------------
### 框架概述
- **主函数接口**  
```
在/scripts中有train.py eval.py demo.py  
train.py：进行模型的训练，请详细阅读流程  
eval.py： 进行模型的预测与评估，这个文件会调用train.py的参数，可选对图像进行上色并保存
demo.py: 进对图像进行预测与上色，可以在这个基础上封装项目
详细的训练命令行实例，可以参考run.txt
```

- **重要文件夹概述**  
```
.{ROOT}
├── core
│   ├── data 读取数据集与下载数据集的脚本，包括voc/citys_fine/city_coarse/等
│   ├── models 模型的加载脚本，重点查看model_zoo.py 目前只修改了psp
│   ├── nn 模型与数据集需要调用的编译脚本，一般情况不要动
│   ├── utils 损失与评估等功能性脚本，主要看损失与评估以及上色部分
├── scripts
│   ├── demo.py 
│   ├── eval.py
│   ├── eval.py
│   ├── eval 模型预测生成结果的保存路径
│   └── models 模型训练保存的快照
```


### 训练
-----------------
- **单卡训练**
```
# 单卡训练例子： Networks: resnet50+psp , dataset: citys_fine
python train.py --model psp --backbone resnet50 --dataset citys_fine --lr 0.01 --batch-size 8 --epochs 120 --devices 9
```
- **Multi-GPU training**

```
# 多卡训练例子，Networks: resnet50+psp , dataset: citys_fine:
export NGPUS=3 
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model psp --backbone resnet50 --dataset citys --lr 0.01 --batch-size 8 --epochs 1 --devices 5,6,7 
```

### 预测与评估
-----------------
```
# 预测例子： Networks: resnet50+psp , dataset: citys_fine
python eval.py --model psp --backbone resnet50 --dataset citys_fine
```
### 预测demo
```
python demo.py
```
## Support

#### Model

- [FCN](https://arxiv.org/abs/1411.4038)
- [ENet](https://arxiv.org/pdf/1606.02147)
- [PSPNet](https://arxiv.org/pdf/1612.01105)
- [ICNet](https://arxiv.org/pdf/1704.08545)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [EncNet](https://arxiv.org/abs/1803.08904v1)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [OCNet](https://arxiv.org/pdf/1809.00916)
- [CGNet](https://arxiv.org/pdf/1811.08201)
- [ESPNetv2](https://arxiv.org/abs/1811.11431)
- [CCNet](https://arxiv.org/pdf/1811.11721)
- [DUNet(DUpsampling)](https://arxiv.org/abs/1903.02120)
- [FastFCN(JPU)](https://arxiv.org/abs/1903.11816)
- [LEDNet](https://arxiv.org/abs/1905.02423)
- [Fast-SCNN](https://github.com/Tramac/Fast-SCNN-pytorch)
- [LightSeg](https://github.com/Tramac/Lightweight-Segmentation)
- [DFANet](https://arxiv.org/abs/1904.02216)

[DETAILS](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md) for model & backbone.
```
.{SEG_ROOT}
├── core
│   ├── models
│   │   ├── bisenet.py
│   │   ├── danet.py
│   │   ├── deeplabv3.py
│   │   ├── deeplabv3+.py
│   │   ├── denseaspp.py
│   │   ├── dunet.py
│   │   ├── encnet.py
│   │   ├── fcn.py
│   │   ├── pspnet.py
│   │   ├── icnet.py
│   │   ├── enet.py
│   │   ├── ocnet.py
│   │   ├── ccnet.py
│   │   ├── psanet.py
│   │   ├── cgnet.py
│   │   ├── espnet.py
│   │   ├── lednet.py
│   │   ├── dfanet.py
│   │   ├── ......
```

#### 数据集

数据集已经下载到服务器，主要请修改数据集读取并加入数据增强

|                           Dataset                            | training set | validation set | testing set |
| :----------------------------------------------------------: | :----------: | :------------: | :---------: |
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) |     1464     |      1449      |      ✘      |
| [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |    11355     |      2857      |      ✘      |
| [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |    20210     |      2000      |      ✘      |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/)  |     2975     |      500       |      ✘      |
| [COCO](http://cocodataset.org/#download)           |              |                |             |
| [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip) |     4085     |      638       |      ✘      |
| [LIP(Look into Person)](http://sysu-hcp.net/lip/)       |    30462     |     10000      |    10000    |

```
.{SEG_ROOT}
├── core
│   ├── data
│   │   ├── dataloader
│   │   │   ├── ade.py
│   │   │   ├── cityscapes.py
│   │   │   ├── mscoco.py
│   │   │   ├── pascal_aug.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── sbu_shadow.py
│   │   └── downloader
│   │       ├── ade20k.py
│   │       ├── cityscapes.py
│   │       ├── mscoco.py
│   │       ├── pascal_voc.py
│   │       └── sbu_shadow.py
```

## Result
- **PASCAL VOC 2012**

|Methods|Backbone|TrainSet|EvalSet|crops_size|epochs|JPU|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|FCN32s|vgg16|train|val|480|60|✘|47.50|85.39|
|FCN16s|vgg16|train|val|480|60|✘|49.16|85.98|
|FCN8s|vgg16|train|val|480|60|✘|48.87|85.02|
|FCN32s|resnet50|train|val|480|50|✘|54.60|88.57|
|PSPNet|resnet50|train|val|480|60|✘|63.44|89.78|
|DeepLabv3|resnet50|train|val|480|60|✘|60.15|88.36|

Note: `lr=1e-4, batch_size=4, epochs=80`.

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```

## To Do
- [x] add train script
- [ ] remove syncbn
- [ ] train & evaluate
- [x] test distributed training
- [x] fix syncbn ([Why SyncBN?](https://tramac.github.io/2019/04/08/SyncBN/))
- [x] add distributed ([How DIST?](https://tramac.github.io/2019/04/22/%E5%88%86%E5%B8%83%E5%BC%8F%E8%AE%AD%E7%BB%83-PyTorch/))

## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
-->

[python-image]: https://img.shields.io/badge/Python-3.7.7-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.6-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[cuda-image]: https://img.shields.io/badge/CUDA-10.2-blue.svg
[cuda-url]: https://developer.nvidia.com/
[torchvision-image]: https://img.shields.io/badge/torchvision-0.6-green.svg
[torchvision-url]: https://pytorch.org/docs/stable/torchvision/index.html/
[tensorflow-image]: https://img.shields.io/badge/tensorflow-2.1-yellow.svg
[tensorflow-url]: https://tensorflow.google.cn/
