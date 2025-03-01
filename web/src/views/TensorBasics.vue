<template>
  <el-card class="tensor-basics-card">
    <template #header>
      <div class="card-header">
        <h2>张量基础</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="张量创建" name="creation">
          <div class="section">
            <h3>张量创建方法</h3>
            <el-card class="code-card">
              <pre><code class="python">{{ tensorCreationCode }}</code></pre>
              <el-button type="primary" @click="runTensorCreation">运行代码</el-button>
            </el-card>
            <div v-if="creationResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ creationResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="张量操作" name="operations">
          <div class="section">
            <h3>基本操作</h3>
            <el-card class="code-card">
              <pre><code class="python">{{ tensorOperationsCode }}</code></pre>
              <el-button type="primary" @click="runTensorOperations">运行代码</el-button>
            </el-card>
            <div v-if="operationsResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ operationsResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="可视化" name="visualization">
          <div class="section">
            <h3>张量可视化</h3>
            <div ref="tensorChart" class="chart-container"></div>
            <el-form :model="visualizationForm" label-width="120px">
              <el-form-item label="张量维度">
                <el-select v-model="visualizationForm.dimensions" placeholder="选择维度">
                  <el-option label="1维" value="1" />
                  <el-option label="2维" value="2" />
                </el-select>
              </el-form-item>
              <el-form-item>
                <el-button type="primary" @click="generateVisualization">生成可视化</el-button>
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

const activeTab = ref('creation')
const creationResult = ref('')
const operationsResult = ref('')
const tensorChart = ref(null)
const chart = ref(null)

const visualizationForm = ref({
  dimensions: '1'
})

const tensorCreationCode = `import torch

# 从列表创建张量
tensor1 = torch.tensor([1, 2, 3, 4, 5])

# 创建全零张量
zeros = torch.zeros(3, 3)

# 创建全一张量
ones = torch.ones(2, 4)

# 创建随机张量
random = torch.rand(2, 2)

print("Tensor1:", tensor1)
print("Zeros:", zeros)
print("Ones:", ones)
print("Random:", random)`

const tensorOperationsCode = `import torch

# 创建两个张量
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 加法
print("加法:", a + b)

# 乘法
print("乘法:", a * b)

# 矩阵乘法
c = torch.tensor([[1, 2], [3, 4]])
d = torch.tensor([[5, 6], [7, 8]])
print("矩阵乘法:", torch.matmul(c, d))`

const runTensorCreation = () => {
  // 模拟运行结果
  creationResult.value = `Tensor1: tensor([1, 2, 3, 4, 5])
Zeros: tensor([[0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]])
Ones: tensor([[1., 1., 1., 1.],
       [1., 1., 1., 1.]])
Random: tensor([[0.1234, 0.5678],
        [0.9012, 0.3456]])`
}

const runTensorOperations = () => {
  // 模拟运行结果
  operationsResult.value = `加法: tensor([5, 7, 9])
乘法: tensor([4, 10, 18])
矩阵乘法: tensor([[19, 22],
        [43, 50]])`
}

const generateVisualization = () => {
  if (!chart.value) return

  const option = {
    title: {
      text: `${visualizationForm.value.dimensions}维张量可视化`
    },
    tooltip: {},
    xAxis: {},
    yAxis: {},
    series: []
  }

  if (visualizationForm.value.dimensions === '1') {
    const data = Array.from({length: 5}, (_, i) => [i, Math.random()])
    option.series.push({
      type: 'line',
      data: data,
      name: '张量值'
    })
  } else {
    const data = Array.from({length: 25}, () => Math.random())
    option.series.push({
      type: 'heatmap',
      data: data.map((value, index) => [
        Math.floor(index / 5),
        index % 5,
        value
      ]),
      name: '张量值'
    })
  }

  chart.value.setOption(option)
}

onMounted(() => {
  // 初始化代码高亮
  document.querySelectorAll('pre code').forEach((block) => {
    hljs.highlightBlock(block)
  })

  // 初始化图表
  if (tensorChart.value) {
    chart.value = echarts.init(tensorChart.value)
    generateVisualization()
  }
})
</script>

<style scoped>
.tensor-basics-card {
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
  width: 100%;
  height: 400px;
  margin: 20px 0;
}
</style>