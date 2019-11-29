import torch
import time
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import random


# 3.1.2.2 矢量计算表达式:同时对多个样本进行矢量计算
def f31():
    a = torch.ones(1000)
    b = torch.ones(1000)
    start_time = time.time()
    c = torch.zeros(1000)
    for i in range(1000):
        c[i] = a[i] + b[i]
    print(time.time() - start_time)
    start_time2 = time.time()
    d = a + b
    print(time.time() - start_time2)


# 3.2 线性回归实现
# 思路: 生成数据集->读取数据->定义模型->初始化模型参数->定义损失函数->定义优化算法->训练模型->测试模型
# 1. 生成数据集
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


# 2读取数据
# 每次返回数据中batch_size个随机样本的特征和标签
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    """
    range(start,stop,step)
    range(5)=range(0,5)=range(0,5,1)=[0,1,2,3,4]
    indices[i:min(i+batch_size,num_examples)]# 当min()返回num_examples时，返回样本数不足batch_size
    yield迭代器:每次返回数据从上次位置迭代不重复,累计返回
    """
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i:min(i + batch_size, num_examples)])
        yield features.index_select(0, j), labels.index_select(0, j)


# 定义模型
def linreg(x, w, b):
    return torch.mm(x, w) + b  # 矩阵乘法


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


# 从0实现线性回归模型
def f32():
    num_inputs = 2  # 输入特征个数
    num_example = 1000  # 训练数据样本数
    true_w = [2, -3.4]  # 真实权重
    true_b = 4.2  # 真实偏差b
    features = torch.randn(num_example, num_inputs)  # 填入size
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    # np.random.normal 正态分布生成随机数
    labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)
    set_figsize()
    plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
    batch_size = 10
    for (x, y) in data_iter(batch_size, features, labels):
        print(x, y)
        break  # 返回一个batch_size
    # 初始化模型参数
    w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)  # 列矩阵[w1,w2]^T
    b = torch.zeros(1, dtype=torch.float32)  # 一个参数b
    # 学习参数-需要求梯度
    w.requires_grad_(True)
    b.requires_grad_(True)
    # 训练模型
    lr = 0.03
    num_epochs = 3  # 学习周期:已一个周期遍历一次整个数据
    for epoch in range(num_epochs):
        for x, y in data_iter(batch_size=batch_size, features=features, labels=labels):
            loss = squared_loss(linreg(x, w, b), y).sum()  # 累计batch_size的loss
            # 反向传播求梯度
            loss.backward()
            # 跟新参数[w,b]
            sgd([w, b], lr, batch_size)
            w.grad.data.zero_()
            b.grad.data.zero_()
        # 每一周期学习
        # 损失值
        train_loss = squared_loss(linreg(features, w, b), labels)
        print('epoch: %d,loss: %f' % (epoch + 1, train_loss.mean().item()))
    # 进行预测
    print(true_w, w)
    print(true_b, b)


