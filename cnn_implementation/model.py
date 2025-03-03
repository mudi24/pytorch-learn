import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层: 输入通道=3 (RGB图像), 输出通道=16, 卷积核大小=3x3
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        # 第一个池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第二个卷积层: 输入通道=16, 输出通道=32, 卷积核大小=3x3
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # 第二个池化层
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第三个卷积层: 输入通道=32, 输出通道=64, 卷积核大小=3x3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 第三个池化层
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层: 假设输入图像为224x224，经过三次池化后为28x28
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout层，防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # 第一个卷积块
        x = self.pool1(F.relu(self.conv1(x)))
        
        # 第二个卷积块
        x = self.pool2(F.relu(self.conv2(x)))
        
        # 第三个卷积块
        x = self.pool3(F.relu(self.conv3(x)))
        
        # 展平操作，将三维特征图转换为一维向量
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_feature_maps(self, x):
        """获取各层的特征图，用于可视化"""
        feature_maps = []
        
        # 第一个卷积层特征图
        x1 = F.relu(self.conv1(x))
        feature_maps.append(x1)
        x = self.pool1(x1)
        
        # 第二个卷积层特征图
        x2 = F.relu(self.conv2(x))
        feature_maps.append(x2)
        x = self.pool2(x2)
        
        # 第三个卷积层特征图
        x3 = F.relu(self.conv3(x))
        feature_maps.append(x3)
        
        return feature_maps