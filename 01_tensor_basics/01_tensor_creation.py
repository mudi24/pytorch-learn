import torch
import numpy as np

def main():
    # 1. 直接从数据创建张量
    print("\n1. 从数据直接创建张量:")
    data = [[1, 2], [3, 4]]
    x = torch.tensor(data)
    print(f"从列表创建的张量:\n{x}")
    print(f"张量的形状: {x.shape}")
    print(f"张量的数据类型: {x.dtype}")

    # 2. 从NumPy数组创建张量
    print("\n2. 从NumPy数组创建张量:")
    np_array = np.array([[1, 2], [3, 4]])
    x_np = torch.from_numpy(np_array)
    print(f"从NumPy数组创建的张量:\n{x_np}")

    # 3. 创建特殊张量
    print("\n3. 创建特殊张量:")
    # 创建全为0的张量
    zeros = torch.zeros(2, 3)
    print(f"全0张量:\n{zeros}")
    
    # 创建全为1的张量
    ones = torch.ones(2, 3)
    print(f"全1张量:\n{ones}")
    
    # 创建未初始化的张量
    empty = torch.empty(2, 3)
    print(f"未初始化张量:\n{empty}")

    # 4. 指定数据类型创建张量
    print("\n4. 指定数据类型创建张量:")
    float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    int_tensor = torch.tensor([1, 2], dtype=torch.int64)
    print(f"浮点型张量: {float_tensor}, 数据类型: {float_tensor.dtype}")
    print(f"整型张量: {int_tensor}, 数据类型: {int_tensor.dtype}")

    # 5. 创建随机张量
    print("\n5. 创建随机张量:")
    # 均匀分布
    rand_uniform = torch.rand(2, 3)
    print(f"均匀分布随机张量:\n{rand_uniform}")
    
    # 标准正态分布
    rand_normal = torch.randn(2, 3)
    print(f"正态分布随机张量:\n{rand_normal}")

    # 6. 创建序列张量
    print("\n6. 创建序列张量:")
    # 使用arange
    arange = torch.arange(0, 10, step=2)
    print(f"等差数列张量: {arange}")
    
    # 使用linspace
    linspace = torch.linspace(0, 10, steps=5)
    print(f"线性等分张量: {linspace}")

if __name__ == "__main__":
    main()