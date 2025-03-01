import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.layers(x)

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.targets = torch.sum(self.data, dim=1, keepdim=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def setup(rank, world_size):
    # 初始化进程组
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    # 清理进程组
    dist.destroy_process_group()

def train(rank, world_size):
    print(f"在进程 {rank} 上运行训练")
    setup(rank, world_size)

    # 创建模型和移动到对应设备
    model = SimpleModel().to(rank)
    # 包装模型用于分布式训练
    model = DDP(model, device_ids=[rank])

    # 创建数据集和分布式采样器
    dataset = SimpleDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0

        for data, targets in dataloader:
            data, targets = data.to(rank), targets.to(rank)
            
            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        # 只在主进程上打印信息
        if rank == 0:
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    cleanup()

def main():
    # 1. 设置进程数
    print("\n1. 初始化分布式训练设置:")
    world_size = 2  # 使用2个进程
    print(f"使用 {world_size} 个进程进行分布式训练")

    # 2. 启动多进程训练
    print("\n2. 启动分布式训练:")
    mp.spawn(
        train,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()