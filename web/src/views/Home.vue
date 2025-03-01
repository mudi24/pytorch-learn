<template>
  <el-card class="welcome-card">
    <template #header>
      <div class="card-header">
        <h2>欢迎使用 PyTorch 学习可视化平台</h2>
      </div>
    </template>
    <div class="content">
      <p class="description">这个平台旨在帮助你系统地学习 PyTorch 的核心概念，从基础知识到高级应用，采用渐进式学习方法。通过交互式的可视化界面，你可以更直观地理解每个概念。</p>
      
      <h3>学习路径</h3>
      <el-timeline>
        <el-timeline-item
          v-for="(item, index) in learningPath"
          :key="index"
          :type="item.type"
          :color="item.color"
          :timestamp="item.title">
          {{ item.content }}
        </el-timeline-item>
      </el-timeline>

      <h3>学习资源</h3>
      <el-row :gutter="20" class="resource-links">
        <el-col :span="8" v-for="(resource, index) in resources" :key="index">
          <el-card shadow="hover" class="resource-card">
            <h4>{{ resource.title }}</h4>
            <el-button type="primary" @click="openLink(resource.url)" text>
              访问链接
              <el-icon class="el-icon--right"><Link /></el-icon>
            </el-button>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </el-card>
</template>

<script setup>
import { Link } from '@element-plus/icons-vue'

const learningPath = [
  {
    title: '张量基础',
    content: '学习 PyTorch 的基本数据结构 - 张量，以及如何进行张量操作',
    type: 'primary',
    color: '#409EFF'
  },
  {
    title: '自动微分',
    content: '理解 PyTorch 的自动微分机制，这是深度学习的核心',
    type: 'success',
    color: '#67C23A'
  },
  {
    title: '神经网络',
    content: '学习如何使用 PyTorch 构建神经网络模型',
    type: 'warning',
    color: '#E6A23C'
  },
  {
    title: '训练过程',
    content: '掌握模型训练、评估和保存的完整流程',
    type: 'danger',
    color: '#F56C6C'
  },
  {
    title: '进阶模型',
    content: '学习构建卷积神经网络(CNN)和循环神经网络(RNN)',
    type: 'info',
    color: '#909399'
  },
  {
    title: '迁移学习',
    content: '学习如何利用预训练模型',
    type: 'primary',
    color: '#409EFF'
  }
]

const resources = [
  {
    title: 'PyTorch 官方文档',
    url: 'https://pytorch.org/docs/stable/index.html'
  },
  {
    title: 'PyTorch 官方教程',
    url: 'https://pytorch.org/tutorials/'
  },
  {
    title: 'PyTorch 论坛',
    url: 'https://discuss.pytorch.org/'
  }
]

const openLink = (url) => {
  window.open(url, '_blank')
}
</script>

<style scoped>
.welcome-card {
  margin-bottom: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.description {
  font-size: 16px;
  line-height: 1.6;
  color: #606266;
  margin-bottom: 30px;
}

.content h3 {
  margin: 30px 0 20px;
  font-weight: 600;
  color: #303133;
}

.resource-links {
  margin-top: 20px;
}

.resource-card {
  text-align: center;
  height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
}

.resource-card h4 {
  margin: 0 0 15px;
  color: #303133;
}
</style>