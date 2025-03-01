import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        # 定义输出层
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden):
        # 前向传播
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden
    
    def init_hidden(self, batch_size):
        # 初始化隐藏状态
        return torch.zeros(1, batch_size, self.hidden_size)

def generate_sequence(length):
    # 生成sin波形序列
    x = np.linspace(0, 10, length)
    sequence = np.sin(x)
    return sequence

def prepare_data(sequence, seq_length):
    # 准备训练数据
    x, y = [], []
    for i in range(len(sequence) - seq_length):
        x.append(sequence[i:i + seq_length])
        y.append(sequence[i + seq_length])
    return torch.FloatTensor(x).view(-1, seq_length, 1), torch.FloatTensor(y).view(-1, 1)

def main():
    # 1. 设置参数
    print("\n1. 初始化参数:")
    input_size = 1  # 输入维度
    hidden_size = 32  # 隐藏层大小
    output_size = 1  # 输出维度
    seq_length = 10  # 序列长度
    num_epochs = 100  # 训练轮数
    learning_rate = 0.01  # 学习率
    
    # 2. 准备数据
    print("\n2. 准备数据:")
    sequence = generate_sequence(1000)
    x, y = prepare_data(sequence, seq_length)
    print(f"输入数据形状: {x.shape}")
    print(f"目标数据形状: {y.shape}")
    
    # 3. 创建模型
    print("\n3. 创建模型:")
    model = SimpleRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(model)
    
    # 4. 训练模型
    print("\n4. 开始训练:")
    for epoch in range(num_epochs):
        model.train()
        hidden = model.init_hidden(x.size(0))
        
        # 前向传播
        outputs, hidden = model(x, hidden)
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # 5. 测试模型
    print("\n5. 测试模型:")
    model.eval()
    with torch.no_grad():
        test_seq = torch.FloatTensor(sequence[:seq_length]).view(1, seq_length, 1)
        hidden = model.init_hidden(1)
        predictions = []
        
        # 生成预测序列
        for _ in range(50):
            output, hidden = model(test_seq, hidden)
            pred = output[0, -1, 0].item()
            predictions.append(pred)
            test_seq = torch.cat((test_seq[:, 1:, :], 
                                output[:, -1:, :]), dim=1)
    
    print("预测序列的前10个值:")
    print(predictions[:10])
    print("\n实际序列的前10个值:")
    print(sequence[seq_length:seq_length+10])

if __name__ == "__main__":
    main()