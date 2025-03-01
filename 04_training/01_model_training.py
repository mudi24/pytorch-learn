import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 创建一个简单的数据集
class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        # 生成随机数据
        self.X = torch.randn(size, 2)
        # 生成标签：如果x1 + x2 > 0，则为1，否则为0
        self.y = (self.X.sum(dim=1) > 0).float().view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
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
        predicted = (outputs > 0.5).float()
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    
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
            
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
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
    # 创建训练集和验证集
    train_dataset = SimpleDataset(size=800)
    val_dataset = SimpleDataset(size=200)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    print("数据加载器创建完成")

    # 3. 创建模型、损失函数和优化器
    print("\n3. 初始化模型:")
    model = SimpleModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print(model)

    # 4. 训练循环
    print("\n4. 开始训练:")
    epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # 验证阶段
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
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
            }, 'best_model.pth')
            print("保存最佳模型")
        print()

    # 5. 加载最佳模型并进行最终测试
    print("\n5. 加载最佳模型并测试:")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = validate_model(model, val_loader, criterion, device)
    print(f'最终测试结果 - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%')

if __name__ == "__main__":
    main()