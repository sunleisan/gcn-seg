# 动手学习-PyTorch[官方教程](http://pytorch123.com/)
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torchvision as tv
import torchvision.transforms as ts
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim


# 入门
def f1():
    # 5*3空矩阵,返回类型:tensor
    e1 = t.empty(5, 3)
    r1 = t.rand(5, 3)  # 0-1随机数
    z1 = t.zeros((5, 3), dtype=t.long)
    t1 = t.tensor([5.5, 3])
    # 基于已有张量创新新张量
    t1 = t.randn_like(t1, dtype=t.float)  # 取结构

    # 操作
    x = t.rand(5, 3)
    y = t.rand(5, 3)
    print(x + y)
    print(t.add(x, y))
    result = t.empty(5, 3)
    t.add(x, y, out=result)  # 输出结果
    # 张量本身会发生变化的操作会有下划线_
    x.t_()
    # 改变大小
    x = t.rand(4, 4)
    y = x.view(16)
    z = x.view(-1, 8)  # -1表示取决于其他维度
    # 获取值
    print(e1, r1, z1, t1, y, z)
    print(x.item())


# 自动微分
def f2():
    x = t.ones(2, 2, requires_grad=True)  # True,变量发生变化将自动求导保留
    print(x)
    y = x + 2
    print(y)
    z = y * y * 3
    out = z.mean()
    print(z, out)
    out.backward()  # 向后传播,手动向后求导
    print(x.grad)  # d(out)/dx求得梯度


# 神经网络
# 定义神经网络
class Net(nn.Module):

    def __init__(self):
        # 调用init函数
        super(Net, self).__init__()
        # 定义卷积运算
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义线性变换y=Ax+b: in_features:输入特征,out_features输出特征,bias:True设置变差b
        # 转换为矩阵运算,x*120(输入)与120*84进行矩阵相乘输出x*84
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 向前传播,神经网络的结构
        # 最大池化(2*2)运算,con1激活值
        x = f.max_pool2d(f.relu(self.conv1(x)), (2, 2))
        # 2维矩阵如果是方阵,可以参数可以简化为一个
        x = f.max_pool2d(f.relu(self.conv2(x)), 2)
        # 把输入x变成输入图片的一维数量
        x = x.view(-1, self.num_flat_features(x))
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        # size:原始值比如(100,28,28,3)输入100个28*28*3个图片
        size = x.size()[1:]  # 除了0索引(batch数量维度)的所有维度
        num_feature = 1
        # s遍历28,28,3
        for s in size:
            num_feature *= s
        # 返回28*28*3每个输入的特征数
        return num_feature


"""
模型可视化
input(28, 28, 1)
conv(24,24,6) # 6个 5*5卷积核
pool(23,23,6) # 2*2最大池化
conv(19,19,16) # 16个 5*5卷积核
pool(18,18,16) # 2*2最大池化
fullyconn(1,1,120) # 全连接
fullyconn(1,1,84) # 全连接
softmax(1,1,10) # 逻辑回归预测
"""


# 模型应用
def f3():
    net = Net()
    print(net)
    # 模型可训练的参数
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())
    # 随机生成模型32*32的输入
    net_input = t.randn(1, 1, 32, 32)
    out = net(net_input)
    print(out)
    # 参数梯度缓存器设置0，随机的梯度反向传播
    net.zero_grad()
    out.backward(t.randn(1, 10))
    # 计算损失
    output = net(net_input)
    target = t.randn(10)  # 模拟10个输入值的准确值
    target = target.view(1, -1)  # 使size与模型输出一致
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)
    # 反向传播,计算相关联参数的梯度
    net.zero_grad()
    print(net.conv1.bias.grad)
    loss.backward()
    print(net.conv1.bias.grad)
    # 跟新参数,随机梯度下降
    learning_rate = 0.01
    for p in net.parameters():
        p.data.sub_(p.grad.data * learning_rate)


# 处理数据
# 数据处理可以使用一些数据处理包
# 图像:OpenCV 语音:scipy,librosa 文本:NLTK,SpaCy
# 计算机视觉提供torchvision包
# CIFAR10数据集: 32*32*3的图片,有10个分类
# 训练一个图片分类器
# 步骤:加载并归一化数据->定义卷积神经网络->定义损失函数->训练网络->测试网络
# PILImage:Python一个图像处理库
# torchvision数据集输出是[0-255]之间的PILImage,转换为[-1,1]的张量
def get_data():
    # 定义变化规则
    transform = ts.Compose(
        [ts.ToTensor(),
         ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # 下载数据集,应用转换规则,第一次运行download=True,后续运行,download=False
    trainset = tv.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    testset = tv.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


# 展示一些训练数据
# 图片数据类型转换
# trainloader()
def imshow(images, labels):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # get some random training images
    img = tv.utils.make_grid(images)
    img = img / 2 + 0.5
    nping = img.numpy()
    plt.imshow(np.transpose(nping, (1, 2, 0)))
    plt.show()
    # print labels
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 定义神经网络
# 6@5*5C-> 2*2P->16@5*5-> 120F->84F->10F
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # in_channels:3,彩色图像,out_channels:6,kernel_size 5*5
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1(x)))
        x = self.pool(f.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train():
    net = Net2()
    # CrossEntropyLoss(weight=None,size_average=True)
    # [手册](https://pytorch-cn.readthedocs.io/zh/latest/package_references/torch-nn/)
    """
    > 监督学习两大问题:分类问题(离散目标变量),回归问题(连续目标变量)
    * 对于回归问题神经网络主要输出一个节点,节点值就是预测值,损失函数常用均方误差
    * 分类问题,不同节点的不同，使用交叉熵Cross Entropy

    [参考](https://blog.csdn.net/xg123321123/article/details/80781611)
    """
    # [梯度下降](https://www.cnblogs.com/lliuye/p/9451903.html)
    """
    > PyTorch常用的优化器
    下降: 如果函数有n个变量(可以学习的参数),沿着梯度方向下降,梯度由所有变量的导数组成。
    * BGD 批量梯度下降法
    每一次迭代(参数跟新或下降或损失值减小)都会遍历所有样本
    收敛快,速度慢
    * SGD 随机梯度下降法
    每次迭代使用一个样本进行参数跟新
    速度快,易受噪音数据影响,准确度差
    * MBGD 小批量梯度下降法
    每次使用batch_size个样本进行迭代
    * Momentum 标准动量优化
    使用了动量(Momentum参数)的随机梯度下降算法
    方法: 将上一步的梯度结合参数Momentum(取0.9)用于下一步的跟新
    思路: 上一步梯度大，预测增加下一步下降速度, 上一步梯度小,预测减缓下一步下降速度
    整体上增加了梯度下降的速度.
    * RMSProp
    * Adam
    """
    criterion = nn.CrossEntropyLoss()
    # 随机梯度下降法,可以指定不同层不同学习率,有batch_size,momentum
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # 加载数据
    transform = ts.Compose(
        [ts.ToTensor(),
         ts.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)
    for epoch in range(2):
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 跟新参数
            running_loss += loss.item()  # 累积2000mini-batches损失
            if i % 2000 == 1999:
                print("[%d,%5d] loss:%.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print("Finished Training")
    print("保存网络")
    # t.save(net, 'net.pkl')  # 保存整个网络
    t.save(net.state_dict(), 'net_params.pkl')  # 保存参数


def test():
    # net = t.load('net.pkl')  # 加载整个网络
    net = Net2()
    net.load_state_dict(t.load('net_params.pkl'))
    # 测试样本数据,准备数据
    trainloader, testloader, classes = get_data()
    testiter = iter(testloader)
    images, labels = testiter.next()
    # 显示图片
    print("显示图片")
    imshow(images, labels)
    # 预测数据
    outputs = net(images)  # 输出10个类别概率batch_size=4,有4组输出
    # dim=1,指定输出维度，输出最大值和索引
    _, predicted = t.max(outputs, 1)  # 有4组最大值
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))


# 在整个数据集上测试正确率
def test2():
    net = Net2()
    net.load_state_dict(t.load('net_params.pkl'))
    correct = 0
    total = 0
    trainloader, testloader, classes = get_data()
    with t.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = t.max(outputs.data, 1)
            total += labels.size(0)  # 元素个数=size(0)=batch_size=4
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


"""
如何在GPU上训练模型
cuda:基于GPU的并行计算平台
"""
