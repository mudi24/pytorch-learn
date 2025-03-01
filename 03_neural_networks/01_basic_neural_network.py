import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 定义网络层
        self.layer1 = nn.Linear(2, 4)  # 输入层到隐藏层
        self.activation = nn.ReLU()     # 激活函数
        self.layer2 = nn.Linear(4, 1)   # 隐藏层到输出层
        self.sigmoid = nn.Sigmoid()      # 输出层激活函数

    def forward(self, x):
        # 定义前向传播
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

def main():
    # 1. 创建模型实例
    print("\n1. 创建模型:")
    model = SimpleNN()
    print(model)

    # 2. 准备数据
    print("\n2. 准备数据:")
    # 创建一些示例数据
    X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    print(f"输入数据 X:\n{X}")
    print(f"目标数据 y:\n{y}")

    # 3. 定义损失函数和优化器
    print("\n3. 定义损失函数和优化器:")
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.1)  # 随机梯度下降优化器

    # 4. 训练模型
    print("\n4. 训练模型:")
    epochs = 1000
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每100轮打印一次损失
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    # 5. 测试模型
    print("\n5. 测试模型:")
    with torch.no_grad():
        test_output = model(X)
        print(f"预测结果:\n{test_output.round()}")
        print(f"实际标签:\n{y}")

if __name__ == "__main__":
    main()