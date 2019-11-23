import numpy as np


def basics():
    # 图的邻接矩阵
    A = np.matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 0],
        [1, 0, 1, 0]],
        dtype=float)
    # 提取特征集合:第一个特征索引,第二个特征,索引负值
    # np.shape返回一个数组(n,m)行数和列数
    # 这里A.shape[0]返回值就是图的结点序号0-3
    X = np.matrix([
        [i, -i]
        for i in range(A.shape[0])
    ], dtype=float)
    # 应用传播规则,矩阵相乘
    # 每个节点的特征值将会变为相邻节点的聚合:可以手动验证
    print(A * X)
    # 问题1: 信息的聚合不包括自身(除非有自环)
    # 问题2: 度的大小会影响特征->导致梯度消失或梯度爆炸,影响随机梯度下降算法
    # 网络很深时:多个小于1的数相乘会消失,多个大于1的数相乘会爆炸[参考](https://blog.csdn.net/qq_24502469/article/details/90490233)
    # 方法: 添加自环,特征归一化处理
    I = np.eye(A.shape[0])  # 通过单位矩阵添加自环
    A_hat = A + I
    print(A_hat * X)
    # 归一化方法: D^-1*A*X
    # 度矩阵,axis=0列相加,axis=1行相加,返回二维数组,[0]提取成一维数组
    print(np.sum(A, axis=0))
    D = np.array(np.sum(A, axis=0))[0]  # [1,2,2,1]
    D = np.matrix(np.diag(D))
    # 传播规则
    print(D ** -1 * A * X)  # shape:4*2
    # 整合使用
    D_hat = np.array(np.sum(A_hat, axis=0))[0]
    D_hat = np.matrix(np.diag(D_hat))
    # 权重,不同类型特征的权重,列数代表输出特征的维度(数量)
    W = np.matrix([
        [1, -1],
        [-1, 1]
    ], dtype=float)
    input_feature = D_hat ** -1 * A_hat * X * W
    print(input_feature)
    # 使用激活函数ReLU(start_input)

# 空手道俱乐部应用
# 构建一个GCN,不真正训练该网络, 进行随机初始化
# networkx -> note
