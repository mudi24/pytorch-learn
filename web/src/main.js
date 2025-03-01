import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import App from './App.vue'

// 路由配置
const routes = [
  {
    path: '/',
    component: () => import('./views/Home.vue')
  },
  {
    path: '/tensor',
    component: () => import('./views/TensorBasics.vue')
  },
  {
    path: '/autograd',
    component: () => import('./views/Autograd.vue')
  },
  {
    path: '/neural-networks',
    component: () => import('./views/NeuralNetworks.vue')
  },
  {
    path: '/training',
    component: () => import('./views/Training.vue')
  },
  {
    path: '/cnn',
    component: () => import('./views/CNN.vue')
  },
  {
    path: '/rnn',
    component: () => import('./views/RNN.vue')
  },
  {
    path: '/transfer-learning',
    component: () => import('./views/TransferLearning.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

const app = createApp(App)
app.use(router)
app.use(ElementPlus)
app.mount('#app')