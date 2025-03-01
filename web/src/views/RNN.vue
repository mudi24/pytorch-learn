<template>
  <el-card class="rnn-card">
    <template #header>
      <div class="card-header">
        <h2>循环神经网络</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="网络结构" name="structure">
          <div class="section">
            <h3>RNN结构</h3>
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
        <el-tab-pane label="序列处理" name="sequence">
          <div class="section">
            <h3>序列数据处理</h3>
            <el-form :model="sequenceConfig" label-width="120px">
              <el-form-item label="序列长度">
                <el-input-number v-model="sequenceConfig.seqLength" :min="1" :max="100" />
              </el-form-item>
              <el-form-item label="隐藏层大小">
                <el-input-number v-model="sequenceConfig.hiddenSize" :min="1" :max="256" />
              </el-form-item>
              <el-form-item label="层数">
                <el-input-number v-model="sequenceConfig.numLayers" :min="1" :max="5" />
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="processSequence">处理序列</el-button>
              </el-form-item>
            </el-form>
            <div ref="sequenceChart" class="chart-container"></div>
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
const sequenceChart = ref(null)
const chart = ref(null)

const sequenceConfig = ref({
  seqLength: 10,
  hiddenSize: 64,
  numLayers: 2
})

const networkCode = `import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 创建模型实例
model = RNN(input_size=28, hidden_size=128, num_layers=2, num_classes=10)
print(model)`

const runNetworkExample = () => {
  networkResult.value = `RNN(
  (rnn): RNN(28, 128, num_layers=2, batch_first=True)
  (fc): Linear(in_features=128, out_features=10, bias=True)
)`
}

const processSequence = () => {
  if (!chart.value) return

  // 模拟序列数据处理
  const seqLength = sequenceConfig.value.seqLength
  const hiddenStates = Array.from({length: seqLength}, () => 
    Array.from({length: sequenceConfig.value.hiddenSize}, () => Math.random())
  )

  const option = {
    title: {
      text: '隐藏状态可视化'
    },
    tooltip: {
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      data: Array.from({length: seqLength}, (_, i) => `t${i+1}`)
    },
    yAxis: {
      type: 'value',
      name: '隐藏状态值'
    },
    series: Array.from({length: 5}, (_, i) => ({
      name: `神经元 ${i+1}`,
      type: 'line',
      data: hiddenStates.map(state => state[i]),
      smooth: true
    }))
  }

  chart.value.setOption(option)
}

onMounted(() => {
  // 初始化代码高亮
  document.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightBlock(block)
  })

  // 初始化图表
  if (sequenceChart.value) {
    chart.value = echarts.init(sequenceChart.value)
  }
})
</script>

<style scoped>
.rnn-card {
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
  margin: 20px 0;
  width: 600px;
}
</style>