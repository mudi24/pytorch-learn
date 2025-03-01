import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18

def create_model(num_classes):
    # 加载预训练的ResNet18模型
    model = resnet18(pretrained=True)
    
    # 冻结所有层的参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

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
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 加载CIFAR-10数据集
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    print("数据加载器创建完成")

    # 3. 创建模型
    print("\n3. 初始化模型:")
    model = create_model(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # 只优化最后一层的参数
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    print("模型创建完成，使用预训练的ResNet18")

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
            }, 'best_transfer_model.pth')
            print("保存最佳模型")
        print()

    # 5. 加载最佳模型并进行最终测试
    print("\n5. 加载最佳模型并测试:")
    checkpoint = torch.load('best_transfer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = validate_model(model, test_loader, criterion, device)
    print(f'最终测试结果 - Loss: {final_loss:.4f}, Accuracy: {final_acc:.2f}%')

if __name__ == "__main__":
    main()