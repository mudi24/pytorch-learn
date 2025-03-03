from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import numpy as np
# 修改导入语句
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn_implementation.model import SimpleCNN

app = Flask(__name__)
# 修改 CORS 配置
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# 加载模型
model = SimpleCNN()
try:
    model.load_state_dict(torch.load('./cnn_implementation/models/cnn_model.pth'))
    model.eval()
except:
    print("未找到预训练模型，使用随机初始化的模型")

def preprocess_image(image_data):
    # 将 base64 图片数据转换为 PIL Image
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # 定义预处理转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # 预处理图片
    return transform(image).unsqueeze(0)

@app.route('/api/cnn/process', methods=['POST'])
def process_image():
    try:
        # 获取上传的图片
        image_data = request.json['image']
        
        # 预处理图片
        input_tensor = preprocess_image(image_data)
        
        # 获取特征图
        with torch.no_grad():
            feature_maps = model.get_feature_maps(input_tensor)
        
        # 将特征图转换为可以JSON序列化的格式
        processed_maps = []
        for layer_maps in feature_maps:
            # 只取第一个特征图作为示例
            feature_map = layer_maps[0, 0].cpu().numpy()
            processed_maps.append(feature_map.tolist())
        
        return jsonify({
            'status': 'success',
            'feature_maps': processed_maps
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)