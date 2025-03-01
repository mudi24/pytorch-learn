<template>
  <el-card class="training-card">
    <template #header>
      <div class="card-header">
        <h2>模型训练</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="训练配置" name="config">
          <div class="section">
            <h3>训练参数配置</h3>
            <el-form :model="trainingConfig" label-width="120px">
              <el-form-item label="优化器">
                <el-select v-model="trainingConfig.optimizer">
                  <el-option label="SGD" value="sgd" />
                  <el-option label="Adam" value="adam" />
                  <el-option label="RMSprop" value="rmsprop" />
                </el-select>
              </el-form-item>
              <el-form-item label="学习率">
                <el-slider v-model="trainingConfig.learningRate" :min="0.0001" :max="0.1" :step="0.0001" />
              </el-form-item>
              <el-form-item label="批次大小">
                <el-input-number v-model="trainingConfig.batchSize" :min="1" :max="256" />
              </el-form-item>
              <el-form-item label="训练轮数">
                <el-input-number v-model="trainingConfig.epochs" :min="1" :max="100" />
              </el-form-item>
              <el-form-item label="损失函数">
                <el-select v-model="trainingConfig.lossFunction">
                  <el-option label="交叉熵损失" value="cross_entropy" />
                  <el-option label="均方误差损失" value="mse" />
                  <el-option label="L1损失" value="l1" />
                </el-select>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>
        <el-tab-pane label="训练过程" name="process">
          <div class="section">
            <h3>训练监控</h3>
            <div ref="trainingChart" class="chart-container"></div>
            <div class="control-panel">
              <el-button type="primary" @click="startTraining">开始训练</el-button>
              <el-button type="warning" @click="pauseTraining">暂停</el-button>
              <el-button type="danger" @click="stopTraining">停止</el-button>
            </div>
          </div>
        </el-tab-pane>
      </el-tabs>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import * as echarts from 'echarts'

const activeTab = ref('config')
const trainingChart = ref(null)
const chart = ref(null)

const trainingConfig = ref({
  optimizer: 'adam',
  learningRate: 0.001,
  batchSize: 32,
  epochs: 10,
  lossFunction: 'cross_entropy'
})

const startTraining = () => {
  if (!chart.value) return

  // 模拟训练过程数据
  const epochs = trainingConfig.value.epochs
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

const pauseTraining = () => {
  // 实现暂停训练逻辑
}

const stopTraining = () => {
  // 实现停止训练逻辑
}

onMounted(() => {
  if (trainingChart.value) {
    chart.value = echarts.init(trainingChart.value)
  }
})
</script>

<style scoped>
.training-card {
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

.chart-container {
  width: 600px;
  height: 400px;
  margin: 20px 0;
}

.control-panel {
  margin-top: 20px;
  display: flex;
  gap: 10px;
}
</style>