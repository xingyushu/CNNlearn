# CNNlearn
基于经典cnn结构构建的图片分类系统_人工智能 与机器学习课堂作业
主要做了两项工作
1. 利用tensorflow对CIFAR-10 数据集的分类，包含数据集处理，训练，生成模型和测试
(1)配置好python+tensorflow即可
  可以使用下列命令获取源码
git clone https://github.com/tensorflow/models.git
cd models/tutorials/image/cifar10

cifar10_input.py	读取本地CIFAR-10的二进制文件格式的内容。
cifar10.py	        建立CIFAR-10的模型。
cifar10_train.py	在CPU或GPU上训练CIFAR-10的模型。
cifar10_multi_gpu_train.py	在多GPU上训练CIFAR-10的模型。
cifar10_eval.py	    评估CIFAR-10模型的预测性能。
具体操作 可以参考链接查看中文版的介绍  http://www.tensorfly.cn/tfdoc/tutorials/deep_cnn.html

(2)下载数据集：
 binary格式：http://www.cs.toronto.edu/~kriz/cifar.html
 图片格式：https://pan.baidu.com/s/1skN4jW5   z6i3
 
2. Resnet网络的构建和使用
下载imagenet2012数据集 https://www.cnblogs.com/zjutzz/p/6083201.html
tensorflow-resnet模型： https://github.com/ry/tensorflow-resnet
其它参考：https://github.com/xvshu/ImageNet-Api  

3. PyQt的使用
  使用教程 http://code.py40.com/pyqt5/32.html





