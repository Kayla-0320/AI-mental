import { useState } from 'react';
import { Badge, Tooltip, Button, Popover } from 'antd';
import {
  ExperimentOutlined, KeyOutlined, CameraOutlined,
  ThunderboltOutlined, CloseOutlined,
} from '@ant-design/icons';
import { useAnxiety } from '../context/AnxietyContext';

const levelColors = { low: '#52c41a', medium: '#faad14', high: '#ff4d4f' };
const levelLabels = { low: '良好', medium: '轻度', high: '偏高' };

export default function AnxietyFloatingWidget({ compact = false }: { compact?: boolean }) {
  const { state, startKeyboard, stopKeyboard, startBlink, stopBlink } = useAnxiety();
  const [expanded, setExpanded] = useState(false);

  const { combinedIndex, level, keyboardActive, blinkActive } = state;

  if (compact) {
    return (
      <Tooltip title={`焦虑指数: ${combinedIndex} (${levelLabels[level]})`}>
        <div
          onClick={() => setExpanded(!expanded)}
          style={{
            position: 'fixed', bottom: 80, right: 20, zIndex: 1000,
            width: 44, height: 44, borderRadius: 22,
            background: levelColors[level],
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', boxShadow: '0 2px 12px rgba(0,0,0,0.15)',
            transition: 'all 0.3s',
          }}
        >
          <ExperimentOutlined style={{ color: '#fff', fontSize: 18 }} />
        </div>
      </Tooltip>
    );
  }

  return (
    <>
      {/* 浮动按钮 */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          position: 'fixed', bottom: 80, right: 20, zIndex: 1000,
          width: 48, height: 48, borderRadius: 24,
          background: `linear-gradient(135deg, ${levelColors[level]}, ${levelColors[level]}dd)`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', boxShadow: '0 4px 16px rgba(0,0,0,0.2)',
          transition: 'all 0.3s',
        }}
      >
        <Badge count={combinedIndex > 50 ? combinedIndex : 0} size="small" offset={[4, -4]}>
          <ExperimentOutlined style={{ color: '#fff', fontSize: 20 }} />
        </Badge>
      </div>

      {/* 展开面板 */}
      {expanded && (
        <div style={{
          position: 'fixed', bottom: 140, right: 20, zIndex: 1000,
          width: 280, background: '#fff', borderRadius: 16,
          boxShadow: '0 8px 32px rgba(0,0,0,0.12)', padding: 16,
          border: `2px solid ${levelColors[level]}30`,
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <span style={{ fontWeight: 600, fontSize: 14 }}>焦虑感知</span>
            <CloseOutlined onClick={() => setExpanded(false)} style={{ cursor: 'pointer', color: '#999' }} />
          </div>

          {/* 综合指数 */}
          <div style={{ textAlign: 'center', marginBottom: 12 }}>
            <div style={{ fontSize: 36, fontWeight: 'bold', color: levelColors[level] }}>{combinedIndex}</div>
            <div style={{ fontSize: 12, color: levelColors[level] }}>{levelLabels[level]}</div>
          </div>

          {/* 键盘控制 */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 0', borderTop: '1px solid #f0f0f0',
          }}>
            <div>
              <KeyOutlined style={{ marginRight: 6, color: '#1890ff' }} />
              <span style={{ fontSize: 13 }}>键盘节奏</span>
              {keyboardActive && <span style={{ fontSize: 11, color: '#52c41a', marginLeft: 6 }}>监测中</span>}
            </div>
            <Button
              size="small" type={keyboardActive ? 'default' : 'primary'}
              onClick={keyboardActive ? stopKeyboard : startKeyboard}
            >
              {keyboardActive ? '停止' : '开始'}
            </Button>
          </div>

          {/* 眨眼控制 */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 0', borderTop: '1px solid #f0f0f0',
          }}>
            <div>
              <CameraOutlined style={{ marginRight: 6, color: '#722ed1' }} />
              <span style={{ fontSize: 13 }}>眨眼检测</span>
              {blinkActive && <span style={{ fontSize: 11, color: '#52c41a', marginLeft: 6 }}>监测中</span>}
            </div>
            <Button
              size="small" type={blinkActive ? 'default' : 'primary'}
              onClick={blinkActive ? stopBlink : startBlink}
            >
              {blinkActive ? '停止' : '开始'}
            </Button>
          </div>

          {/* 详细数据 */}
          {(keyboardActive || blinkActive) && (
            <div style={{ marginTop: 8, padding: 8, background: '#fafafa', borderRadius: 8, fontSize: 11 }}>
              {keyboardActive && (
                <div style={{ marginBottom: 4 }}>
                  键盘: {state.keyboard.typingSpeed}字/分 | 删除{state.keyboard.deletionRate}% | 焦虑{state.keyboard.anxietyIndex}
                </div>
              )}
              {blinkActive && (
                <div>
                  眨眼: {state.blink.blinkRate}次/分 | 焦虑{state.blink.anxietyIndex}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </>
  );
}
