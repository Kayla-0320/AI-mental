import { Component, ErrorInfo, ReactNode } from 'react';
import { Button, Result } from 'antd';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('ErrorBoundary caught:', error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fff5f7' }}>
          <Result
            status="error"
            title="页面出错了"
            subTitle="抱歉，页面加载时发生了错误，请尝试刷新页面。"
            extra={
              <Button type="primary" onClick={this.handleReset} style={{ borderRadius: 24 }}>
                重试
              </Button>
            }
          />
        </div>
      );
    }

    return this.props.children;
  }
}
