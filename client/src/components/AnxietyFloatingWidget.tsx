/**
 * 多模态情绪感知浮动小组件
 *
 * 设计理念：温暖、治愈、适合青少年用户
 * - 圆润卡片 + 柔和渐变
 * - 情绪用自然/天气隐喻（而非冰冷的数字）
 * - 强调"被看见"的温暖感
 */
import { useState, useEffect } from 'react';
import { Badge, Tooltip, Switch, Progress, Modal, Button } from 'antd';
import {
  ExperimentOutlined, CloseOutlined, EyeOutlined,
  SoundOutlined, SmileOutlined, KeyOutlined,
  VideoCameraOutlined,
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
  const { comprehensiveState: state, cameraEnabled, toggleCamera, circadianMetrics, cognitiveMetrics, hrvMetrics, breathingMetrics, behavioralMetrics, eyeMetrics, voiceSemanticsMetrics } = useAnxiety();
  const [expanded, setExpanded] = useState(false);
  const [showPermissionModal, setShowPermissionModal] = useState(false);

  // 首次进入时弹出权限引导（仅一次，用 localStorage 记录）
  useEffect(() => {
    const hasSeen = localStorage.getItem('multimodal_permission_shown');
    if (!hasSeen && !cameraEnabled) {
      const timer = setTimeout(() => {
        setShowPermissionModal(true);
      }, 3000); // 进入 3 秒后弹出
      return () => clearTimeout(timer);
    }
  }, []);

  const handleEnablePermission = async () => {
    setShowPermissionModal(false);
    localStorage.setItem('multimodal_permission_shown', '1');
    if (!cameraEnabled) {
      await toggleCamera();
    }
  };

  const handleDismissPermission = () => {
    setShowPermissionModal(false);
    localStorage.setItem('multimodal_permission_shown', '1');
  };

  const { dominantEmotion, confidence, riskLevel, activeModalities, emotionProbs, caringMessage, facial, voice, keyboard, text, circadian, cognitiveDistortion, hrv, breathing, behavioralActivation, eyeMovement, voiceSemantics } = state;
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
      {/* 首次权限引导弹窗 */}
      <Modal
        open={showPermissionModal}
        onCancel={handleDismissPermission}
        footer={null}
        width={400}
        centered
      >
        <div style={{ textAlign: 'center', padding: '16px 0' }}>
          <div style={{ fontSize: 48, marginBottom: 12 }}>🌈</div>
          <div style={{ fontSize: 18, fontWeight: 600, color: '#4a4a6a', marginBottom: 8 }}>
            开启多模态情绪感知
          </div>
          <div style={{
            padding: '12px 16px', background: '#f8f6ff', borderRadius: 12,
            textAlign: 'left', marginBottom: 16, fontSize: 13, color: '#666', lineHeight: 1.8,
          }}>
            <div>📹 <strong>摄像头</strong> → 面部微表情分析（FACS 动作单元）</div>
            <div>🔊 <strong>麦克风</strong> → 语音声学特征（语调、语速、停顿）</div>
            <div>⌨️ <strong>键盘</strong> → 打字节奏分析（已自动开启）</div>
            <div style={{ marginTop: 8, fontSize: 11, color: '#999' }}>
              🔒 所有数据仅在你的设备上分析，不会上传到云端。你可以随时关闭。
            </div>
          </div>
          <div style={{ display: 'flex', gap: 12 }}>
            <Button block size="large" onClick={handleDismissPermission} style={{ borderRadius: 12 }}>
              稍后再说
            </Button>
            <Button block size="large" type="primary" onClick={handleEnablePermission}
              style={{ borderRadius: 12, background: '#6366f1' }}>
              <VideoCameraOutlined /> 开启感知
            </Button>
          </div>
        </div>
      </Modal>

      {/* 全站感知状态条（开启时在顶部显示） */}
      {cameraEnabled && (
        <div style={{
          position: 'fixed', top: 0, left: 0, right: 0, zIndex: 999,
          height: 3, background: 'linear-gradient(90deg, #6366f1, #a78bfa, #6366f1)',
          backgroundSize: '200% 100%',
          animation: 'shimmer 2s ease-in-out infinite',
        }}>
          <style>{`@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; }`}</style>
        </div>
      )}

      {/* 浮动按钮 */}
      <div
        onClick={() => setExpanded(!expanded)}
        style={{
          width: 48, height: 48, borderRadius: '50%',
          background: `linear-gradient(135deg, ${emotionColor}30, ${emotionColor}15)`,
          border: `2px solid ${emotionColor}25`,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          cursor: 'pointer', fontSize: 22,
          boxShadow: `0 4px 20px ${emotionColor}20`,
          transition: 'all 0.3s ease',
          flexShrink: 0,
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
          width: 300, maxHeight: 'calc(100vh - 170px)', overflowY: 'auto',
          background: 'linear-gradient(180deg, #fefefe 0%, #f8f6ff 100%)',
          borderRadius: 20, boxShadow: '0 8px 40px rgba(99,102,241,0.15)',
          padding: 16, border: `1.5px solid ${emotionColor}15`,
          pointerEvents: 'auto',
        }}>
          {/* 标题栏 */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
            <span style={{ fontWeight: 600, fontSize: 14, color: '#4a4a6a' }}>
              🌈 你的情绪天气
            </span>
            <CloseOutlined onClick={() => setExpanded(false)} style={{ cursor: 'pointer', color: '#bbb', fontSize: 12 }} />
          </div>

          {/* 开关 */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '8px 12px', background: cameraEnabled ? 'linear-gradient(135deg, #f0f9ff, #f5f3ff)' : '#fafafa',
            borderRadius: 12, marginBottom: 10,
            border: `1px solid ${cameraEnabled ? '#d6bcfa30' : '#f0f0f0'}`,
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 18 }}>{cameraEnabled ? '👁️' : '💤'}</span>
              <div>
                <div style={{ fontSize: 12, fontWeight: 500, color: '#4a4a6a' }}>
                  {cameraEnabled ? '正在温柔地感知你' : '开启多模态感知'}
                </div>
                <div style={{ fontSize: 10, color: '#999' }}>
                  {cameraEnabled ? `11 模态融合 · 活跃 ${activeModalities.length} 个` : '文字 · 声音 · 面部 · 键盘 · 生理 · 认知'}
                </div>
              </div>
            </div>
            <Switch
              checked={cameraEnabled}
              onChange={toggleCamera}
              checkedChildren="开"
              unCheckedChildren="关"
              style={{ background: cameraEnabled ? '#6366f1' : undefined }}
              size="small"
            />
          </div>

          {/* 情绪天气展示 */}
          {cameraEnabled && (
            <>
              <div style={{ textAlign: 'center', padding: '4px 0 10px' }}>
                <div style={{ fontSize: 36, marginBottom: 2, lineHeight: 1.2 }}>{weather.icon}</div>
                <div style={{ fontSize: 16, fontWeight: 600, color: emotionColor }}>{weather.text}</div>
                <div style={{ fontSize: 11, color: '#999', marginTop: 2 }}>
                  置信度 {Math.round(confidence * 100)}%
                </div>
              </div>

              {/* 五维情绪分布 */}
              <div style={{ marginBottom: 10 }}>
                {['快乐', '悲伤', '焦虑', '愤怒', '中性'].map((label, i) => {
                  const prob = emotionProbs[i] || 0;
                  const color = EMOTION_COLORS[label];
                  return (
                    <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                      <span style={{ fontSize: 11, color: '#888', width: 30, textAlign: 'right' }}>{label}</span>
                      <div style={{ flex: 1, height: 5, background: '#f0f0f0', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{
                          height: '100%', width: `${Math.round(prob * 100)}%`,
                          background: `linear-gradient(90deg, ${color}80, ${color})`,
                          borderRadius: 3, transition: 'width 0.8s ease',
                        }} />
                      </div>
                      <span style={{ fontSize: 10, color, fontWeight: 500, width: 30 }}>{Math.round(prob * 100)}%</span>
                    </div>
                  );
                })}
              </div>

              {/* 11 模态三层架构指示器 */}
              <div style={{ marginBottom: 8 }}>
                {/* 基础感知层 */}
                <div style={{ marginBottom: 4 }}>
                  <div style={{ fontSize: 9, color: '#999', marginBottom: 2, fontWeight: 500 }}> 基础感知</div>
                  <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                    {[
                      { active: text.timestamp > 0, icon: '📝', label: '文字', color: '#1890ff' },
                      { active: voice.isRecording, icon: '🔊', label: '语音', color: '#52c41a' },
                      { active: facial.isDetecting, icon: '😊', label: '面部', color: '#fa8c16' },
                      { active: keyboard.isCollecting, icon: '⌨️', label: '键盘', color: '#722ed1' },
                    ].map((m, i) => (
                      <div key={i} style={{
                        padding: '2px 6px',
                        background: m.active ? `${m.color}15` : '#f5f5f5',
                        borderRadius: 5,
                        fontSize: 9,
                        color: m.active ? m.color : '#bbb',
                        display: 'flex', alignItems: 'center', gap: 2,
                        border: `1px solid ${m.active ? `${m.color}30` : 'transparent'}`,
                        transition: 'all 0.3s ease',
                      }}>
                        <span>{m.icon}</span>
                        <span>{m.label}</span>
                        {m.active && <span style={{ width: 3, height: 3, borderRadius: 2, background: m.color }} />}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 认知分析层 */}
                <div style={{ marginBottom: 4 }}>
                  <div style={{ fontSize: 9, color: '#999', marginBottom: 2, fontWeight: 500 }}>🧠 认知分析</div>
                  <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                    {[
                      { active: circadian?.isActive || circadianMetrics?.isActive, icon: '🕐', label: '昼夜', color: '#13c2c2' },
                      { active: (cognitiveDistortion?.riskScore || cognitiveMetrics?.riskScore || 0) > 0, icon: '💭', label: '认知', color: '#eb2f9a' },
                      { active: (voiceSemantics?.riskScore || voiceSemanticsMetrics?.riskScore || 0) > 0 || (voiceSemantics?.timestamp || voiceSemanticsMetrics?.timestamp || 0) > 0, icon: '🗣️', label: '语义', color: '#faad14' },
                    ].map((m, i) => (
                      <div key={i} style={{
                        padding: '2px 6px',
                        background: m.active ? `${m.color}15` : '#f5f5f5',
                        borderRadius: 5,
                        fontSize: 9,
                        color: m.active ? m.color : '#bbb',
                        display: 'flex', alignItems: 'center', gap: 2,
                        border: `1px solid ${m.active ? `${m.color}30` : 'transparent'}`,
                        transition: 'all 0.3s ease',
                      }}>
                        <span>{m.icon}</span>
                        <span>{m.label}</span>
                        {m.active && <span style={{ width: 3, height: 3, borderRadius: 2, background: m.color }} />}
                      </div>
                    ))}
                  </div>
                </div>

                {/* 生理行为层 */}
                <div>
                  <div style={{ fontSize: 9, color: '#999', marginBottom: 2, fontWeight: 500 }}>💓 生理行为</div>
                  <div style={{ display: 'flex', gap: 3, flexWrap: 'wrap' }}>
                    {[
                      { active: hrv?.isMeasuring || hrvMetrics?.isMeasuring, icon: '❤️', label: '心率', color: '#ff4d4f' },
                      { active: breathing?.isMeasuring || breathingMetrics?.isMeasuring, icon: '🌬️', label: '呼吸', color: '#36cfc9' },
                      { active: behavioralActivation?.isActive || behavioralMetrics?.isActive, icon: '🏃', label: '行为', color: '#597ef7' },
                      { active: eyeMovement?.isMeasuring || eyeMetrics?.isMeasuring, icon: '👁️', label: '眼动', color: '#9254de' },
                    ].map((m, i) => (
                      <div key={i} style={{
                        padding: '2px 6px',
                        background: m.active ? `${m.color}15` : '#f5f5f5',
                        borderRadius: 5,
                        fontSize: 9,
                        color: m.active ? m.color : '#bbb',
                        display: 'flex', alignItems: 'center', gap: 2,
                        border: `1px solid ${m.active ? `${m.color}30` : 'transparent'}`,
                        transition: 'all 0.3s ease',
                      }}>
                        <span>{m.icon}</span>
                        <span>{m.label}</span>
                        {m.active && <span style={{ width: 3, height: 3, borderRadius: 2, background: m.color }} />}
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* 温暖治愈消息 */}
              <div style={{
                padding: '8px 12px', background: 'linear-gradient(135deg, #fff5f5, #fff0f6)',
                borderRadius: 12, textAlign: 'center', marginBottom: 6,
              }}>
                <div style={{ fontSize: 12, color: '#8a6d9a', lineHeight: 1.5 }}>
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
