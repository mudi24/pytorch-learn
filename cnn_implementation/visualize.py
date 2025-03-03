import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import SimpleCNN

def load_image(image_path, img_size=224):
    """加载并预处理图像"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 添加批次维度
    
    return image, image_tensor

def visualize_feature_maps(model, image_tensor, layer_idx=0, num_features=8):
    """可视化特定层的特征图"""
    # 确保模型处于评估模式
    model.eval()
    
    # 获取特征图
    with torch.no_grad():
        feature_maps = model.get_feature_maps(image_tensor)
    
    # 选择指定层的特征图
    if layer_idx < len(feature_maps):
        feature_map = feature_maps[layer_idx]
    else:
        print(f"层索引 {layer_idx} 超出范围，最大索引为 {len(feature_maps)-1}")
        return
    
    # 将特征图转换为numpy数组
    feature_map = feature_map.squeeze(0).cpu().numpy()
    
    # 确定要显示的特征图数量
    num_features = min(num_features, feature_map.shape[0])
    
    # 创建图形
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    
    # 显示特征图
    for i in range(num_features):
        ax = axes[i]
        ax.imshow(feature_map[i], cmap='viridis')
        ax.set_title(f'特征 {i+1}')
        ax.axis('off')
    
    # 隐藏未使用的子图
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'./feature_maps_layer_{layer_idx+1}.png')
    plt.show()

def main():
    # 加载模型
    model = SimpleCNN(num_classes=10)
    
    try:
        # 尝试加载预训练权重
        model.load_state_dict(torch.load('./models/cnn_model.pth'))
        print("已加载预训练模型")
    except:
        print("未找到预训练模型，使用随机初始化的模型")
    
    # 加载图像
    image_path = input("请输入图像路径: ")
    original_image, image_tensor = load_image(image_path)
    
    # 显示原始图像
    plt.figure(figsize=(6, 6))
    plt.imshow(original_image)
    plt.title("输入图像")
    plt.axis('off')
    plt.show()
    
    # 可视化每一层的特征图
    for i in range(3):  # 我们的模型有3个卷积层
        print(f"可视化第 {i+1} 个卷积层的特征图")
        visualize_feature_maps(model, image_tensor, layer_idx=i)

if __name__ == "__main__":
    main()