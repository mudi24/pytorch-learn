<template>
  <el-card class="neural-networks-card">
    <template #header>
      <div class="card-header">
        <h2>神经网络</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="网络结构" name="structure">
          <div class="section">
            <h3>神经网络结构</h3>
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
        <el-tab-pane label="训练过程" name="training">
          <div class="section">
            <h3>训练过程监控</h3>
            <div ref="trainingChart" class="chart-container"></div>
            <el-form :model="trainingForm" label-width="120px">
              <el-form-item label="学习率">
                <el-slider v-model="trainingForm.learningRate" :min="0.001" :max="0.1" :step="0.001" />
              </el-form-item>
              <el-form-item label="批次大小">
                <el-input-number v-model="trainingForm.batchSize" :min="1" :max="128" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="startTraining">开始训练</el-button>
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
const trainingChart = ref(null)
const chart = ref(null)

const trainingForm = ref({
  learningRate: 0.01,
  batchSize: 32
})

const networkCode = `import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()
print(model)`

const runNetworkExample = () => {
  networkResult.value = `SimpleNet(
  (fc1): Linear(in_features=784, out_features=128, bias=True)
  (relu): ReLU()
  (fc2): Linear(in_features=128, out_features=10, bias=True)
)`
}

const startTraining = () => {
  if (!chart.value) return

  // 模拟训练过程数据
  const epochs = 10
  const trainLoss = Array.from({length: epochs}, () => Math.random() * 0.5)
  const valLoss = Array.from({length: epochs}, () => Math.random() * 0.7)

  const option = {
    title: {
      text: '训练过程监控'
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['训练损失', '验证损失']
    },
    xAxis: {
      type: 'category',
      data: Array.from({length: epochs}, (_, i) => `Epoch ${i+1}`)
    },
    yAxis: {
      type: 'value',
      name: 'Loss'
    },
    series: [
      {
        name: '训练损失',
        type: 'line',
        data: trainLoss,
        smooth: true
      },
      {
        name: '验证损失',
        type: 'line',
        data: valLoss,
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
  if (trainingChart.value) {
    chart.value = echarts.init(trainingChart.value)
    startTraining()
  }
})
</script>

<style scoped>
.neural-networks-card {
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
  width: 100%;
  margin: 20px 0;
  min-width: 600px;
}
</style>