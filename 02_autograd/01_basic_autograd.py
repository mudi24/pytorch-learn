import torch

def main():
    # 1. 创建需要追踪梯度的张量
    print("\n1. 创建需要追踪梯度的张量:")
    x = torch.tensor([2.0], requires_grad=True)
    print(f"x = {x}")
    print(f"是否需要梯度: {x.requires_grad}")

    # 2. 进行一些运算
    print("\n2. 进行一些运算:")
    y = x * 2
    z = y * y + 1
    print(f"y = x * 2 = {y}")
    print(f"z = y * y + 1 = {z}")

    # 3. 计算梯度
    print("\n3. 计算梯度:")
    z.backward()
    print(f"x的梯度 (dz/dx): {x.grad}")

    # 4. 梯度累积和清零
    print("\n4. 梯度累积和清零:")
    # 再次进行反向传播
    y = x * 2
    z = y * y + 1
    z.backward()
    print(f"累积后的梯度: {x.grad}")
    
    # 清零梯度
    x.grad.zero_()
    print(f"清零后的梯度: {x.grad}")

    # 5. 停止梯度追踪
    print("\n5. 停止梯度追踪:")
    with torch.no_grad():
        y = x * 2
        print(f"y是否需要梯度: {y.requires_grad}")

    # 6. 分离张量
    print("\n6. 分离张量:")
    y = x * 2
    y_detached = y.detach()
    print(f"原张量是否需要梯度: {y.requires_grad}")
    print(f"分离后的张量是否需要梯度: {y_detached.requires_grad}")

if __name__ == "__main__":
    main()