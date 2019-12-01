> 计划安排
* [gcn-简单实现](https://mp.weixin.qq.com/s/sg9O761F0KHAmCPOfMW_kQ)OK
* 完成PyTorch入门教程 OK
* 完成curve-gcn工程运行 OK
* 完成动手学习深度学习PyTorch版本线性回归部分后续部分持续学习 OK
* 完成PyTorch官方教程
> pytorch
[官网](https://pytorch.org/)
[中文官方手册](https://pytorch-cn.readthedocs.io/zh/latest/)
[视频教程-莫烦Python](https://morvanzhou.github.io/tutorials/machine-learning/torch/)
[视频教程-莫烦Python-BiliBili](https://www.bilibili.com/video/av15997678/?p=1)
[pyTorch学习资料](https://github.com/INTERMT/Awesome-PyTorch-Chinese)
[pyTorch官方教程](http://pytorch123.com/FirstSection/InstallIutorial/#12)
[动手学习深度学习-PyTorch-github](https://github.com/ShusenTang/Dive-into-DL-PyTorch)
[动手学习深度学习-PyTorch-网页](https://tangshusen.me/Dive-into-DL-PyTorch/#/)
> 图卷积神经网络
[PyTorch-GCN-代码](https://github.com/tkipf/pygcn)
[PyTorch-GCN-简单代码](https://github.com/johncava/GCN-pytorch)
> Jupyter
[参考](http://baijiahao.baidu.com/s?id=1601883438842526311&wfr=spider&for=pc)
> 数据问题
> 训练问题
[Kaggle](https://www.cnblogs.com/lvdongjie/p/11435363.html)
[google-cala](https://blog.csdn.net/dfql83704/article/details/101359496)
> 编程问题
1. 可视化工具
(1)代码可视化成模型
(2)绘制模型得工具
[3D可视化](https://blog.csdn.net/c9yv2cf9i06k2a9e/article/details/84038702)
[latax在线工具](https://www.overleaf.com/)
[参考](https://blog.csdn.net/WZZ18191171661/article/details/87886588)
> Python相关
[self用法](https://www.cnblogs.com/wangjian941118/p/9360471.html)
self代表类的实例
> 学习记录
* PyTorch官方教程
# 2019.11.25
1. nn.Conv2d
[参考](https://blog.csdn.net/qq_26369907/article/details/88366147)
2. 手写数字识别三维可视化工具
[可视化](http://scs.ryerson.ca/~aharley/vis/conv/)
3. 辅助代码绘制模型工具
[编码结合模型工具](https://cbovar.github.io/ConvNetDraw/)
弄清楚代码
可视化代码模型
4. [LeNet的三维模型手写数字识别](https://tensorspace.org/html/playground/lenet_zh.html)
5. 神经网络建模工具
* 建模工具
[灵活简洁工具](http://alexlenail.me/NN-SVG/LeNet.html)
[编码进行建模](https://cbovar.github.io/ConvNetDraw/)
* 参考模型
[手写数字识别三维可视化模型](http://scs.ryerson.ca/~aharley/vis/conv/)
[LeNet的三维模型手写数字识别](https://tensorspace.org/html/playground/lenet_zh.html)
6. 迭代器
[参考](https://www.cnblogs.com/wangcoo/p/10018363.html)
是一个对象:用于访问集合的元素
特点: 可以记住遍历的位置
可迭代对象: 数组，列表等
7. DataLoader迭代时出错:BrokenPipeError: [Errno 32] Broken pipe
[参考](https://blog.csdn.net/qq_33666011/article/details/81873217)
# 11.27
8. 不同损失函数总结
9. Softmax
[参考](https://blog.csdn.net/bitcarmanlee/article/details/82320853)
Softmax是一个可以将输入映射成0-1之间的数,并且保证输出和为1的函数
可以用于神经网络多分类来输出不同分类的取得概率
互斥关系的多分类输出应该使用K个二分类器
训练: 使用最大似然思路,输入值x的分类是y,则输出y的概率应尽可能大,则-y的概率应该尽可能小,构造损失函数-log(output_y)用于训练叫做softmax-loss
## 2019.12.1
10. Python基本数据格式
11. [*,**使用](https://blog.csdn.net/yilovexing/article/details/80577510)
可以用于传参:*传元组,**传字典
