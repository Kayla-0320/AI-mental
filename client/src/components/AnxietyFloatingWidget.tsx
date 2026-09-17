/**
 * 多模态情绪感知浮动小组件
 *
 * 设计理念：温暖、治愈、适合青少年用户
 * - 圆润卡片 + 柔和渐变
 * - 情绪用自然/天气隐喻（而非冰冷的数字）
 * - 强调"被看见"的温暖感
 */
import { useState } from 'react';
import { Badge, Tooltip, Switch, Progress } from 'antd';
import {
  ExperimentOutlined, CloseOutlined, EyeOutlined,
  SoundOutlined, SmileOutlined, KeyOutlined,
} from '@ant-design/icons';
import { useAnxiety } from '../context/AnxietyContext';

const EMOTION_COLORS: Record<string, string> = {
  '快乐': '#52c41a', '悲伤': '#722ed1', '焦虑': '#ff7a45',
  '愤怒': '#ff4d4f', '中性': '#1890ff', '等待感知': '#bbb',
};

// 情绪 → 天气隐喻（青少年友好）
const EMOTION_WEATHER: Record<string, { icon: string; text: string }> = {
  '快乐': { icon: '☀️', text: '阳光明媚' },
  '悲伤': { icon: '🌧️', text: '下着小雨' },
  '焦虑': { icon: '🌬️', text: '有点起风了' },
  '愤怒': { icon: '⛈️', text: '雷电交加' },
  '中性': { icon: '⛅', text: '多云转晴' },
  '等待感知': { icon: '🌱', text: '等待萌芽' },
};

export default function AnxietyFloatingWidget({ compact = false }: { compact?: boolean }) {
  const { comprehensiveState: state, cameraEnabled, toggleCamera } = useAnxiety();
  const [expanded, setExpanded] = useState(false);

  const { dominantEmotion, confidence, riskLevel, activeModalities, emotionProbs, caringMessage, facial, voice, keyboard, text } = state;
  // cameraEnabled 来自 Context 独立状态，不依赖 activeModalities
  const weather = EMOTION_WEATHER[dominantEmotion] || EMOTION_WEATHER['中性'];
  const emotionColor = EMOTION_COLORS[dominantEmotion] || '#1890ff';
  const anxietyIndex = Math.round((emotionProbs[2] || 0) * 100);
  const level = anxietyIndex > 60 ? 'high' : anxietyIndex > 30 ? 'medium' : 'low';
  const levelColors = { low: '#52c41a', medium: '#faad14', high: '#ff7a45' };

  if (compact) {
    return (
      <Tooltip title={`${weather.icon} ${weather.text}`}>
        <div
          onClick={() => setExpanded(!expanded)}
          style={{
            position: 'fixed', bottom: 80, right: 20, zIndex: 1000,
            width: 44, height: 44, borderRadius: 22,
            background: `linear-gradient(135deg, ${levelColors[level]}40, ${levelColors[level]}20)`,
            border: `2px solid ${levelColors[level]}30`,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            cursor: 'pointer', fontSize: 20,
            pointerEvents: 'auto',
          }}
        >
          {weather.icon}
        </div>
      </Tooltip>
    );
  }

  return (
    <>
      {/* 浮动按钮 — pointerEvents:none 避免遮挡底层输入框 */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          position: 'fixed', bottom: 80, right: 20, zIndex: 1000,
          width: 52, height: 52, borderRadius: 26,
          background: `linear-gradient(135deg, ${emotionColor}30, ${emotionColor}15)`,
          border: `2px solid ${emotionColor}25`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', fontSize: 24,
          boxShadow: `0 4px 20px ${emotionColor}20`,
          transition: 'all 0.3s ease',
          pointerEvents: 'auto', // 按钮本身可点击
        }}
      >
        <Badge dot={cameraEnabled} color={levelColors[level]} offset={[-2, -2]}>
          <span>{weather.icon}</span>
        </Badge>
      </div>

      {/* 展开面板 */}
      {expanded && (
        <div style={{
          position: 'fixed', bottom: 145, right: 20, zIndex: 1001,
          width: 320, background: 'linear-gradient(180deg, #fefefe 0%, #f8f6ff 100%)',
          borderRadius: 20, boxShadow: '0 8px 40px rgba(99,102,241,0.15)',
          padding: 20, border: `1.5px solid ${emotionColor}15`,
          pointerEvents: 'auto', // 面板可交互
        }}>
          {/* 标题栏 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
            <span style={{ fontWeight: 600, fontSize: 15, color: '#4a4a6a' }}>
              🌈 你的情绪天气
            </span>
            <CloseOutlined onClick={() => setExpanded(false)} style={{ cursor: 'pointer', color: '#bbb' }} />
          </div>

          {/* 开关 */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '12px 16px', background: cameraEnabled ? 'linear-gradient(135deg, #f0f9ff, #f5f3ff)' : '#fafafa',
            borderRadius: 14, marginBottom: 16,
            border: `1px solid ${cameraEnabled ? '#d6bcfa30' : '#f0f0f0'}`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <span style={{ fontSize: 20 }}>{cameraEnabled ? '👁️' : '💤'}</span>
              <div>
                <div style={{ fontSize: 13, fontWeight: 500, color: '#4a4a6a' }}>
                  {cameraEnabled ? '正在温柔地感知你' : '开启多模态感知'}
                </div>
                <div style={{ fontSize: 11, color: '#999' }}>
                  {cameraEnabled ? `已采集 ${activeModalities.length} 个维度` : '文字 · 声音 · 面部 · 键盘'}
                </div>
              </div>
            </div>
            <Switch
              checked={cameraEnabled}
              onChange={toggleCamera}
              checkedChildren="开"
              unCheckedChildren="关"
              style={{ background: cameraEnabled ? '#6366f1' : undefined }}
            />
          </div>

          {/* 情绪天气展示 */}
          {cameraEnabled && (
            <>
              <div style={{ textAlign: 'center', padding: '8px 0 16px' }}>
                <div style={{ fontSize: 48, marginBottom: 4 }}>{weather.icon}</div>
                <div style={{ fontSize: 18, fontWeight: 600, color: emotionColor }}>{weather.text}</div>
                <div style={{ fontSize: 12, color: '#999', marginTop: 4 }}>
                  置信度 {Math.round(confidence * 100)}%
                </div>
              </div>

              {/* 五维情绪分布 */}
              <div style={{ marginBottom: 16 }}>
                {['快乐', '悲伤', '焦虑', '愤怒', '中性'].map((label, i) => {
                  const prob = emotionProbs[i] || 0;
                  const color = EMOTION_COLORS[label];
                  return (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                      <span style={{ fontSize: 12, color: '#888', width: 36, textAlign: 'right' }}>{label}</span>
                      <div style={{ flex: 1, height: 6, background: '#f0f0f0', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%', width: `${Math.round(prob * 100)}%`,
                          background: `linear-gradient(90deg, ${color}80, ${color})`,
                          borderRadius: 3, transition: 'width 0.8s ease',
                        }} />
                      </div>
                      <span style={{ fontSize: 11, color, fontWeight: 500, width: 36 }}>{Math.round(prob * 100)}%</span>
                    </div>
                  );
                })}
              </div>

              {/* 各模态状态指示 */}
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginBottom: 12 }}>
                {text.timestamp > 0 && (
                  <div style={{ padding: '4px 10px', background: '#f0f9ff', borderRadius: 8, fontSize: 11, color: '#1890ff', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <span>📝</span> 文字感知
                  </div>
                )}
                {voice.isRecording && (
                  <div style={{ padding: '4px 10px', background: '#f6ffed', borderRadius: 8, fontSize: 11, color: '#52c41a', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <SoundOutlined /> 声音感知
                  </div>
                )}
                {facial.isDetecting && (
                  <div style={{ padding: '4px 10px', background: '#fff7e6', borderRadius: 8, fontSize: 11, color: '#fa8c16', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <SmileOutlined /> 面部感知
                  </div>
                )}
                {keyboard.isCollecting && (
                  <div style={{ padding: '4px 10px', background: '#f9f0ff', borderRadius: 8, fontSize: 11, color: '#722ed1', display: 'flex', alignItems: 'center', gap: 4 }}>
                    <KeyOutlined /> 键盘感知
                  </div>
                )}
              </div>

              {/* 温暖治愈消息 */}
              <div style={{
                padding: '12px 16px', background: 'linear-gradient(135deg, #fff5f5, #fff0f6)',
                borderRadius: 14, textAlign: 'center', marginBottom: 8,
              }}>
                <div style={{ fontSize: 13, color: '#8a6d9a', lineHeight: 1.6 }}>
                  {caringMessage}
                </div>
              </div>
            </>
          )}

          {/* 隐私提示 */}
          <div style={{
            padding: '8px 12px', background: '#fffbe620', borderRadius: 10,
            fontSize: 11, color: '#d48806', lineHeight: 1.6, textAlign: 'center',
          }}>
            🔒 所有数据仅在你的设备上分析，加密存储，可随时关闭
          </div>
        </div>
      )}
    </>
  );
}
