<template>
  <el-card class="transfer-learning-card">
    <template #header>
      <div class="card-header">
        <h2>迁移学习</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="模型选择" name="model">
          <div class="section">
            <h3>预训练模型</h3>
            <el-form :model="modelConfig" label-width="120px">
              <el-form-item label="基础模型">
                <el-select v-model="modelConfig.baseModel">
                  <el-option label="ResNet18" value="resnet18" />
                  <el-option label="VGG16" value="vgg16" />
                  <el-option label="MobileNet" value="mobilenet" />
                </el-select>
              </el-form-item>
              <el-form-item label="目标任务">
                <el-select v-model="modelConfig.task">
                  <el-option label="图像分类" value="classification" />
                  <el-option label="目标检测" value="detection" />
                  <el-option label="语义分割" value="segmentation" />
                </el-select>
              </el-form-item>
            </el-form>
            <el-card class="code-card">
              <pre><code class="python">{{ modelCode }}</code></pre>
              <el-button type="primary" @click="runModelExample">运行代码</el-button>
            </el-card>
            <div v-if="modelResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ modelResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="训练对比" name="comparison">
          <div class="section">
            <h3>训练效果对比</h3>
            <el-form :model="trainingConfig" label-width="120px">
              <el-form-item label="微调层数">
                <el-input-number v-model="trainingConfig.fineTuneLayers" :min="1" :max="5" />
              </el-form-item>
              <el-form-item label="学习率">
                <el-slider v-model="trainingConfig.learningRate" :min="0.0001" :max="0.01" :step="0.0001" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="startComparison">开始对比</el-button>
              </el-form-item>
            </el-form>
            <div ref="comparisonChart" class="chart-container"></div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted, watchEffect } from 'vue'
import * as echarts from 'echarts'
import 'highlight.js/styles/github.css'
import hljs from 'highlight.js/lib/core'
import python from 'highlight.js/lib/languages/python'

hljs.registerLanguage('python', python)

const activeTab = ref('model')
const modelResult = ref('')
const comparisonChart = ref(null)
const chart = ref(null)

const modelConfig = ref({
  baseModel: 'resnet18',
  task: 'classification'
})

const trainingConfig = ref({
  fineTuneLayers: 2,
  learningRate: 0.001
})

const modelCode = ref('')

const updateModelCode = () => {
  const baseModel = modelConfig.value.baseModel
  const task = modelConfig.value.task

  // 根据选择的模型和任务生成相应的代码
  const code = `import torch
import torchvision.models as models
import torch.nn as nn

# 加载预训练模型
model = models.${baseModel}(pretrained=True)

# 冻结基础层参数
for param in model.parameters():
    param.requires_grad = False

# 根据任务修改模型结构
${task === 'classification' ? `# 修改最后的全连接层用于分类任务
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)` : 
task === 'detection' ? `# 修改模型用于目标检测任务
from torchvision.models.detection import ${baseModel.includes('resnet') ? 'FasterRCNN' : 'RetinaNet'}

backbone = model.features if hasattr(model, 'features') else nn.Sequential(*list(model.children())[:-2])
model = ${baseModel.includes('resnet') ? 'FasterRCNN' : 'RetinaNet'}(backbone, num_classes=num_classes + 1)` :
`# 修改模型用于语义分割任务
from torchvision.models.segmentation import DeepLabV3

backbone = model.features if hasattr(model, 'features') else nn.Sequential(*list(model.children())[:-2])
model = DeepLabV3(backbone, num_classes=num_classes)`}

print(model)`

  modelCode.value = code
}

// 监听配置变化
watchEffect(() => {
  updateModelCode()
})

const runModelExample = () => {
  const baseModel = modelConfig.value.baseModel
  const task = modelConfig.value.task

  // 根据不同模型和任务生成相应的输出结果
  if (task === 'classification') {
    modelResult.value = `${baseModel.toUpperCase()}(
  (features): Sequential(
    (0): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    ...
  )
  (fc): Linear(in_features=512, out_features=num_classes, bias=True)
)`
  } else if (task === 'detection') {
    modelResult.value = `${baseModel.includes('resnet') ? 'FasterRCNN' : 'RetinaNet'}(
  (backbone): ${baseModel.toUpperCase()}(...)
  (rpn): RegionProposalNetwork(...)
  (roi_heads): RoIHeads(...)
)`
  } else {
    modelResult.value = `DeepLabV3(
  (backbone): ${baseModel.toUpperCase()}(...)
  (classifier): DeepLabHead(
    (aspp): ASPP(...)
    (classifier): Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
  )
)`
  }
}

const startComparison = () => {
  if (!chart.value) return

  // 模拟训练对比数据
  const epochs = 10
  const pretrainedAcc = Array.from({length: epochs}, (_, i) => 0.7 + (Math.random() * 0.2))
  const scratchAcc = Array.from({length: epochs}, (_, i) => 0.3 + (Math.random() * 0.4))

  const option = {
    title: {
      text: '训练效果对比'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['迁移学习', '从头训练']
    },
    xAxis: {
      type: 'category',
      data: Array.from({length: epochs}, (_, i) => `Epoch ${i+1}`)
    },
    yAxis: {
      type: 'value',
      name: '准确率',
      min: 0,
      max: 1
    },
    series: [
      {
        name: '迁移学习',
        type: 'line',
        data: pretrainedAcc,
        smooth: true
      },
      {
        name: '从头训练',
        type: 'line',
        data: scratchAcc,
        smooth: true
      }
    ]
  }

  chart.value.setOption(option)
}

onMounted(() => {
  // 初始化代码高亮
  document.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightBlock(block)
  })

  // 初始化图表
  if (comparisonChart.value) {
    chart.value = echarts.init(comparisonChart.value)
  }
})
</script>

<style scoped>
.transfer-learning-card {
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

.chart-container {
  height: 400px;
  width: 600px;
  margin: 20px 0;
  min-width: 600px;
}
</style>