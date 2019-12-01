# [PyTorch官方教程-强化部分](http://pytorch123.com/ThirdSection/DataLoding/)
# PyTorch之数据加载和处理
# 1.下载安装包
from __future__ import print_function, division
import os
import torch
import pandas as pd  # 用于csv解析
import skimage  # 用于图像io和变换
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision as tv
import warnings
import cv2 as cv  # 利用OpenCV处理图像
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# warnings.filters("ignore")
# plt.ion()  # interactive mode

# 显示图片image及标注点landmarks,图像处理相关使用OpenCV即可
def show_landmarks(image_url, landmarks):
    # 组成带有标记点的图像
    img = cv.imread(image_url)
    # 标记点:图像基底,坐标,半径,颜色,实心
    for i in range(landmarks.shape[0]):
        cv.circle(img, (int(landmarks[i, 0]), int(landmarks[i, 1])), 1, (0, 0, 255), -1)
    # plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r') # 方法二
    # 用于图的跟新过程显示
    # plt.pause(0.001)  # pause a bit so that plots are updated
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


# 数据集类:加载数据,使用图片再加载到内存
class FaceLandmarksDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    # 获取一张图片,item图片索引
    def __getitem__(self, item):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[item, 0])
        image = cv.imread(img_name)
        landmarks = self.landmarks_frame.iloc[item, 1:]
        landmarks = landmarks.astype('float').reshape(-1, 2)
        # 定义数据格式
        sample = {'image': image, 'landmarks': landmarks}
        # 需要变换
        if self.transform:
            sample = self.transform(sample)

        return sample


def show_landmarks2(image, landmarks):
    """显示带有地标的图片"""
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')
    plt.pause(1)  # pause a bit so that plots are updated


def f1():
    # 1 下载数据集
    # 2 读取数据: 从csv文件中读取标注点数据
    # pandas->IO工具DataFrame类型
    landmarks_frame = pd.read_csv('data/faces/faces/face_landmarks.csv')
    n = 10  # 随机一张图片:(数据行索引)
    # 获取;使用iloc按位置获取返回DataFrame类型
    img_name = landmarks_frame.iloc[n, 0]
    # 特征点数据,values->将对象类型转换为数组
    landmarks = landmarks_frame.iloc[n, 1:].values
    # 3 设置float类型,size:两列(x,y)坐标,数组类型
    landmarks = landmarks.astype('float').reshape(-1, 2)
    print('Image name: {}'.format(img_name))
    print('Landmarks shape: {}'.format(landmarks.shape))
    print('First 4 Landmarks: {}'.format(landmarks[:4]))
    # 4 显示一张图片
    # show_landmarks("data/faces/faces/" + img_name, landmarks)
    # 5 数据集类
    face_dataset = FaceLandmarksDataset(csv_file='data/faces/faces/face_landmarks.csv', root_dir="data/faces/faces")
    # 6 数据可视化
    # len,调用__len__
    """
    OpenCV没有集成一个窗口显示多张图片
    """
    for i in range(len(face_dataset)):
        # 调用__getitem__
        sample = face_dataset[i]
        print(i, sample['image'].shape, sample['landmarks'].shape)
        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_landmarks2(**sample)
        if i == 3:
            plt.show()
            break
    # 7 数据变换
    # 8 组合转换
    # 9迭代数据集
    # 10 torchvision


f1()
