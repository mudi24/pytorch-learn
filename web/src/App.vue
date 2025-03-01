<template>
  <el-container class="app-container">
    <el-aside width="250px">
      <el-menu
        router
        :default-active="$route.path"
        class="el-menu-vertical">
        <el-menu-item index="/">
          <el-icon><HomeFilled /></el-icon>
          <span>首页</span>
        </el-menu-item>
        <el-menu-item index="/tensor">
          <el-icon><Grid /></el-icon>
          <span>张量基础</span>
        </el-menu-item>
        <el-menu-item index="/autograd">
          <el-icon><Share /></el-icon>
          <span>自动微分</span>
        </el-menu-item>
        <el-menu-item index="/neural-networks">
          <el-icon><Connection /></el-icon>
          <span>神经网络</span>
        </el-menu-item>
        <el-menu-item index="/training">
          <el-icon><DataLine /></el-icon>
          <span>模型训练</span>
        </el-menu-item>
        <el-menu-item index="/cnn">
          <el-icon><PictureFilled /></el-icon>
          <span>卷积神经网络</span>
        </el-menu-item>
        <el-menu-item index="/rnn">
          <el-icon><Refresh /></el-icon>
          <span>循环神经网络</span>
        </el-menu-item>
        <el-menu-item index="/transfer-learning">
          <el-icon><Switch /></el-icon>
          <span>迁移学习</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header height="60px">
        <h1>PyTorch 学习可视化</h1>
      </el-header>
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import {
  HomeFilled,
  Grid,
  Share,
  Connection,
  DataLine,
  PictureFilled,
  Refresh,
  Switch
} from '@element-plus/icons-vue'
</script>

<style>
.app-container {
  height: 100vh;
}

.el-aside {
  background-color: #304156;
}

.el-menu {
  border-right: none;
}

.el-menu-vertical {
  height: 100%;
  background-color: #304156;
}

.el-menu-item {
  color: #bfcbd9;
}

.el-menu-item.is-active {
  color: #409eff;
  background-color: #263445;
}

.el-header {
  background-color: #fff;
  border-bottom: 1px solid #e6e6e6;
  display: flex;
  align-items: center;
}

.el-main {
  background-color: #f0f2f5;
  padding: 20px;
}
</style>