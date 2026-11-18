import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';
import './index.css';
import './styles/healing-theme.css';
// 首页改版设计系统 + 应用外壳。必须排在 healing-theme.css **之后**：
// 旧主题用 `!important` 写了 body 背景与 .ant-card 样式，同权重时按加载顺序取胜。
import './styles/garden.css';
import './styles/garden-shell.css';
// 全局弹窗排版：顶栏的心理电话 / 多模态 / 反馈，以及首页内容预览、页脚条款弹窗共用
import './styles/garden-modals.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#ffb6c1',
          colorSuccess: '#52c41a',
          colorWarning: '#faad14',
          colorError: '#ff4d4f',
          colorInfo: '#ffb6c1',
          borderRadius: 16,
          colorBgContainer: '#ffffff',
          fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif',
        },
        components: {
          Button: {
            primaryShadow: '0 4px 16px rgba(255,182,193,0.3)',
            algorithm: true,
          },
          Card: {
            borderRadiusLG: 20,
          },
          Modal: {
            borderRadiusLG: 20,
          },
          Input: {
            borderRadius: 12,
            hoverBorderColor: '#ffb6c1',
            activeBorderColor: '#ffb6c1',
            activeShadow: '0 0 0 3px rgba(255,182,193,0.15)',
          },
          Menu: {
            itemBorderRadius: 12,
            itemSelectedBg: '#fff0f3',
            itemSelectedColor: '#ff8fab',
          },
          Tag: {
            borderRadiusSM: 20,
          },
          Collapse: {
            borderRadiusLG: 12,
            headerBg: '#fff5f7',
            contentBg: '#ffffff',
          },
        },
      }}
    >
      <BrowserRouter>
        <App />
      </BrowserRouter>
      </ConfigProvider>
    </ErrorBoundary>
  </React.StrictMode>
);
