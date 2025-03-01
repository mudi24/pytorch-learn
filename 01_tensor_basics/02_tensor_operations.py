import torch

def main():
    # 1. 基本算术运算
    print("\n1. 基本算术运算:")
    a = torch.tensor([[1, 2], [3, 4]])
    b = torch.tensor([[2, 2], [2, 2]])
    
    print(f"张量 a:\n{a}")
    print(f"张量 b:\n{b}")
    print(f"加法 a + b:\n{a + b}")
    print(f"减法 a - b:\n{a - b}")
    print(f"乘法 a * b:\n{a * b}")
    print(f"除法 a / b:\n{a / b}")

    # 2. 矩阵运算
    print("\n2. 矩阵运算:")
    print(f"矩阵乘法 a @ b:\n{a @ b}")
    print(f"矩阵转置 a.T:\n{a.T}")

    # 3. 统计运算
    print("\n3. 统计运算:")
    print(f"求和: {a.sum()}")
    print(f"平均值: {a.mean()}")
    print(f"最大值: {a.max()}")
    print(f"最小值: {a.min()}")
    print(f"按行求和:\n{a.sum(dim=1)}")
    print(f"按列求和:\n{a.sum(dim=0)}")

    # 4. 比较运算
    print("\n4. 比较运算:")
    print(f"a > 2:\n{a > 2}")
    print(f"a < b:\n{a < b}")
    print(f"a == b:\n{a == b}")

    # 5. 形状操作
    print("\n5. 形状操作:")
    c = torch.arange(6)
    print(f"原始张量 c: {c}")
    # 重塑
    reshaped = c.reshape(2, 3)
    print(f"重塑后 (2,3):\n{reshaped}")
    # 视图
    view = c.view(3, 2)
    print(f"视图 (3,2):\n{view}")
    # 增加维度
    unsqueezed = c.unsqueeze(0)
    print(f"增加维度:\n{unsqueezed}")
    # 压缩维度
    squeezed = unsqueezed.squeeze()
    print(f"压缩维度: {squeezed}")

    # 6. 索引和切片
    print("\n6. 索引和切片:")
    d = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"原始张量 d:\n{d}")
    print(f"第一行: {d[0]}")
    print(f"第二列:\n{d[:, 1]}")
    print(f"子矩阵:\n{d[0:2, 1:3]}")

if __name__ == "__main__":
    main()