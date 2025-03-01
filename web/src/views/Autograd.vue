<template>
  <el-card class="autograd-card">
    <template #header>
      <div class="card-header">
        <h2>自动微分</h2>
      </div>
    </template>
    <div class="content">
      <el-tabs v-model="activeTab">
        <el-tab-pane label="基础概念" name="basics">
          <div class="section">
            <h3>自动微分基础</h3>
            <el-card class="code-card">
              <pre><code class="python">{{ basicAutogradCode }}</code></pre>
              <el-button type="primary" @click="runBasicAutograd">运行代码</el-button>
            </el-card>
            <div v-if="basicResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ basicResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="梯度计算" name="gradients">
          <div class="section">
            <h3>梯度计算示例</h3>
            <el-card class="code-card">
              <pre><code class="python">{{ gradientCode }}</code></pre>
              <el-button type="primary" @click="runGradientExample">运行代码</el-button>
            </el-card>
            <div v-if="gradientResult" class="result-card">
              <h4>运行结果：</h4>
              <pre>{{ gradientResult }}</pre>
            </div>
          </div>
        </el-tab-pane>
        <el-tab-pane label="可视化" name="visualization">
          <div class="section">
            <h3>梯度流可视化</h3>
            <div ref="gradientChart" class="chart-container"></div>
            <el-form :model="visualizationForm" label-width="120px">
              <el-form-item label="函数类型">
                <el-select v-model="visualizationForm.function" placeholder="选择函数">
                  <el-option label="二次函数" value="quadratic" />
                  <el-option label="指数函数" value="exponential" />
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

const activeTab = ref('basics')
const basicResult = ref('')
const gradientResult = ref('')
const gradientChart = ref(null)
const chart = ref(null)

const visualizationForm = ref({
  function: 'quadratic'
})

const basicAutogradCode = `import torch

# 创建需要计算梯度的张量
x = torch.tensor([2.0], requires_grad=True)
y = x * 2

# 计算梯度
y.backward()

# 查看梯度
print("x的梯度:", x.grad)`

const gradientCode = `import torch

# 创建输入张量
x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)

# 定义复杂计算
y = x.pow(2).sum()

# 计算梯度
y.backward()

print("输入 x:", x)
print("输出 y:", y)
print("梯度 dx/dy:", x.grad)`

const runBasicAutograd = () => {
  // 模拟运行结果
  basicResult.value = `x的梯度: tensor([2.])`
}

const runGradientExample = () => {
  // 模拟运行结果
  gradientResult.value = `输入 x: tensor([[1., 2.],
        [3., 4.]], requires_grad=True)
输出 y: tensor(30., grad_fn=<SumBackward0>)
梯度 dx/dy: tensor([[2., 4.],
        [6., 8.]])`
}

const generateVisualization = () => {
  if (!chart.value) return

  const x = Array.from({length: 100}, (_, i) => i / 10 - 5)
  let y, dy

  if (visualizationForm.value.function === 'quadratic') {
    y = x.map(x => x * x)
    dy = x.map(x => 2 * x)
  } else {
    y = x.map(x => Math.exp(x))
    dy = x.map(x => Math.exp(x))
  }

  const option = {
    title: {
      text: `${visualizationForm.value.function === 'quadratic' ? '二次' : '指数'}函数及其导数`
    },
    tooltip: {
      trigger: 'axis'
    },
    legend: {
      data: ['函数值', '导数值']
    },
    xAxis: {
      type: 'value',
      name: 'x'
    },
    yAxis: {
      type: 'value',
      name: 'y'
    },
    series: [
      {
        name: '函数值',
        type: 'line',
        data: x.map((x, i) => [x, y[i]]),
        smooth: true
      },
      {
        name: '导数值',
        type: 'line',
        data: x.map((x, i) => [x, dy[i]]),
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
  if (gradientChart.value) {
    chart.value = echarts.init(gradientChart.value)
    generateVisualization()
  }
})
</script>

<style scoped>
.autograd-card {
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
  width: 600px;
  height: 400px;
  margin: 20px 0;
  min-width: 600px;
}
</style>