<template>
  <el-card class="cnn-card">
    <template #header>
      <div class="card-header">
        <h2>卷积神经网络</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="网络结构" name="structure">
          <div class="section">
            <h3>CNN结构</h3>
            <el-card class="code-card">
              <pre><code class="python">{{ networkCode }}</code></pre>
              <el-button type="primary" @click="runNetworkExample">运行代码</el-button>
            </el-card>
            <div v-if="networkResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ networkResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="特征可视化" name="visualization">
          <div class="section">
            <h3>特征图可视化</h3>
            <div class="feature-maps">
              <div v-for="(map, index) in featureMaps" :key="index" class="feature-map">
                <h4>卷积层 {{ index + 1 }}</h4>
                <div :ref="el => { if (el) featureMapRefs[index] = el }" class="map-container"></div>
              </div>
            </div>
            <el-form :model="visualConfig" label-width="120px">
              <el-form-item label="输入图片">
                <el-upload
                  class="image-upload"
                  action="#"
                  :auto-upload="false"
                  :on-change="handleImageChange">
                  <el-button type="primary">选择图片</el-button>
                </el-upload>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="visualizeFeatures">可视化特征图</el-button>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import * as echarts from 'echarts'
import 'highlight.js/styles/github.css'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'

hljs.registerLanguage('python', python)

const activeTab = ref('structure')
const networkResult = ref('')
const featureMaps = ref([])
const featureMapRefs = ref([])

const visualConfig = ref({
  selectedImage: null
})

const networkCode = `import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

# 创建模型实例
model = CNN()
print(model)`

const runNetworkExample = () => {
  networkResult.value = `CNN(
  (conv1): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (relu): ReLU()
  (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0)
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc): Linear(in_features=1568, out_features=10, bias=True)
)`
}

const handleImageChange = (file) => {
  visualConfig.value.selectedImage = file.raw
}

const visualizeFeatures = () => {
  // 模拟特征图数据
  featureMaps.value = Array(2).fill(0).map(() => {
    return Array(16).fill(0).map(() => Array(28).fill(0).map(() => Math.random()))
  })

  // 使用ECharts渲染特征图
  featureMaps.value.forEach((maps, layerIndex) => {
    const chartDom = featureMapRefs.value[layerIndex]
    if (!chartDom) return

    const chart = echarts.init(chartDom)
    chart.setOption({
      tooltip: {
        position: 'top'
      },
      grid: {
        height: '50%',
        top: '10%'
      },
      xAxis: {
        type: 'category',
        data: Array(28).fill(0).map((_, i) => i),
        splitArea: {
          show: true
        }
      },
      yAxis: {
        type: 'category',
        data: Array(28).fill(0).map((_, i) => i),
        splitArea: {
          show: true
        }
      },
      visualMap: {
        min: 0,
        max: 1,
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '15%'
      },
      series: [{
        name: 'Feature Map',
        type: 'heatmap',
        data: maps[0].map((row, i) => 
          row.map((val, j) => [i, j, val])
        ).flat(),
        label: {
          show: false
        },
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }]
    })
  })
}

onMounted(() => {
  // 初始化代码高亮
  document.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightBlock(block)
  })
})
</script>

<style scoped>
.cnn-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.section {
  margin-bottom: 30px;
}

.code-card {
  margin: 20px 0;
  background-color: #f8f9fa;
}

.code-card pre {
  margin: 0;
  padding: 15px;
}

.result-card {
  margin-top: 20px;
  padding: 15px;
  background-color: #f8f9fa;
  border-radius: 4px;
}

.feature-maps {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin: 20px 0;
}

.feature-map {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 15px;
}

.map-container {
  height: 300px;
  margin-top: 10px;
}

.image-upload {
  margin-top: 10px;
}
</style>