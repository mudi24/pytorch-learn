import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 定义卷积层
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 输入通道1（灰度图），输出16通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 28x28 -> 14x14
            
            # 第二个卷积块
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # 16通道 -> 32通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14x14 -> 7x7
        )
        
        # 定义全连接层
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 展平操作
            nn.Linear(32 * 7 * 7, 128),  # 7x7 特征图，32通道
            nn.ReLU(),
            nn.Dropout(0.5),  # 防止过拟合
            nn.Linear(128, 10)  # 10个类别
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算准确率
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    # 1. 设置设备和随机种子
    print("\n1. 初始化设置:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print(f"使用设备: {device}")

    # 2. 准备数据
    print("\n2. 准备数据:")
    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
    ])
    
    # 加载MNIST数据集
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    print("数据加载器创建完成")

    # 3. 创建模型、损失函数和优化器
    print("\n3. 初始化模型:")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print(model)

    # 4. 训练循环
    print("\n4. 开始训练:")
    epochs = 5
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_loss, val_acc = validate_model(model, test_loader, criterion, device)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, 'best_cnn_model.pth')
            print("保存最佳模型")
        print()

    # 5. 加载最佳模型并进行最终测试
    print("\n5. 加载最佳模型并测试:")
    checkpoint = torch.load('best_cnn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = validate_model(model, test_loader, criterion, device)
    print(f'最终测试结果 - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%')

if __name__ == "__main__":
    main()