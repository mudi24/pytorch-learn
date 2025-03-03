import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from model import SimpleCNN
from data_loader import get_data_loaders

def train_model(model, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    """
    训练CNN模型
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 使用学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=2, factor=0.5, verbose=True
    )
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # 检查是否有GPU可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 将模型移动到设备
    model.to(device)
    
    # 训练循环
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 统计
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 计算训练集上的平均损失和准确率
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 计算验证集上的平均损失和准确率
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / val_total
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印统计信息
        time_elapsed = time.time() - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - {time_elapsed:.0f}s')
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}')
        print('-' * 60)
    
    # 保存模型
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(model.state_dict(), './models/cnn_model.pth')
    
    return model, history

if __name__ == "__main__":
    # 创建模型
    model = SimpleCNN(num_classes=10)
    
    # 获取数据加载器
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # 训练模型
    trained_model, history = train_model(
        model, 
        train_loader, 
        test_loader, 
        num_epochs=10
    )
    
    print("模型训练完成!")