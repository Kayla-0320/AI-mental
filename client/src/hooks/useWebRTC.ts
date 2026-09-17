/**
 * WebRTC 双向视频通话 Hook
 *
 * 基于 Socket.IO 信令服务器实现点对点视频连接：
 * 1. 双方加入同一视频房间（按 bookingId 隔离）
 * 2. 发起方创建 RTCPeerConnection → 生成 SDP Offer → 通过信令发送
 * 3. 接收方处理 Offer → 生成 SDP Answer → 通过信令返回
 * 4. 双方交换 ICE Candidate → 建立直连
 * 5. 视频流通过 WebRTC 直连传输（不经过服务器）
 *
 * ICE/STUN/TURN 配置：
 * - 使用 Google 公共 STUN 服务器进行 NAT 穿透
 * - 配置 TURN 服务器作为对称 NAT 的回退方案
 * - 生产环境应部署自有 TURN 服务器（如 coturn）
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import type { Socket } from 'socket.io-client';

// TURN 服务器配置（环境变量优先，回退到免费公共 TURN）
const TURN_USERNAME = import.meta.env.VITE_TURN_USERNAME || 'openrelay';
const TURN_CREDENTIAL = import.meta.env.VITE_TURN_CREDENTIAL || 'openrelay';
const TURN_URL = import.meta.env.VITE_TURN_URL || 'turn:openrelay.metered.ca:80';

const ICE_SERVERS: RTCConfiguration = {
  iceServers: [
    // STUN 服务器（NAT 穿透）
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    // TURN 服务器（对称 NAT 回退）
    {
      urls: TURN_URL,
      username: TURN_USERNAME,
      credential: TURN_CREDENTIAL,
    },
  ],
  iceTransportPolicy: 'all',
  iceCandidatePoolSize: 10,
};

export function useWebRTC(socket: Socket | null, bookingId: string | null) {
  const [localStream, setLocalStream] = useState<MediaStream | null>(null);
  const [remoteStream, setRemoteStream] = useState<MediaStream | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isInitiator, setIsInitiator] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);

  // 创建 RTCPeerConnection
  const createPeerConnection = useCallback(() => {
    const pc = new RTCPeerConnection(ICE_SERVERS);

    // 本地流轨道添加到 PeerConnection
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => {
        pc.addTrack(track, localStreamRef.current!);
      });
    }

    // 接收远端流
    pc.ontrack = (event: RTCTrackEvent) => {
      console.log('[WebRTC] 收到远端媒体流');
      const stream = event.streams[0];
      if (stream) {
        setRemoteStream(stream);
        setIsConnected(true);
      }
    };

    // ICE Candidate 生成 → 通过信令发送给对方
    pc.onicecandidate = (event: RTCPeerConnectionIceEvent) => {
      if (event.candidate && bookingId) {
        socket?.emit('webrtc:ice-candidate', {
          bookingId,
          candidate: event.candidate.toJSON(),
        });
      }
    };

    pc.oniceconnectionstatechange = () => {
      console.log('[WebRTC] ICE 状态:', pc.iceConnectionState);
      if (pc.iceConnectionState === 'connected') {
        setIsConnected(true);
        setError(null);
      } else if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
        setIsConnected(false);
      }
    };

    peerConnectionRef.current = pc;
    return pc;
  }, [socket, bookingId]);

  // 获取本地媒体流
  const startLocalStream = useCallback(async (video = true, audio = true) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: video ? { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } : false,
        audio: audio ? { echoCancellation: true, noiseSuppression: true } : false,
      });
      localStreamRef.current = stream;
      setLocalStream(stream);
      return stream;
    } catch (err) {
      console.error('[WebRTC] 获取媒体流失败:', err);
      setError('无法访问摄像头或麦克风');
      return null;
    }
  }, []);

  // 发起通话（创建 Offer）
  const call = useCallback(async () => {
    if (!socket || !bookingId) return;

    setIsInitiator(true);
    const pc = createPeerConnection();

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    socket.emit('webrtc:offer', {
      bookingId,
      offer: pc.localDescription,
    });

    console.log('[WebRTC] 已发送 Offer');
  }, [socket, bookingId, createPeerConnection]);

  // 接听通话（处理 Offer，创建 Answer）
  const answerCall = useCallback(async () => {
    if (!socket || !bookingId) return;
    const pc = createPeerConnection();
    return pc;
  }, [socket, bookingId, createPeerConnection]);

  // 挂断
  const hangUp = useCallback(() => {
    peerConnectionRef.current?.close();
    peerConnectionRef.current = null;
    localStreamRef.current?.getTracks().forEach((t) => t.stop());
    localStreamRef.current = null;
    setLocalStream(null);
    setRemoteStream(null);
    setIsConnected(false);
    setIsInitiator(false);
    setError(null);

    if (bookingId) {
      socket?.emit('webrtc:leave', bookingId);
    }
  }, [socket, bookingId]);

  // 切换摄像头
  const toggleCamera = useCallback(async (enabled: boolean) => {
    if (!localStreamRef.current) return;
    localStreamRef.current.getVideoTracks().forEach((track) => {
      track.enabled = enabled;
    });
  }, []);

  // 切换麦克风
  const toggleMic = useCallback(async (enabled: boolean) => {
    if (!localStreamRef.current) return;
    localStreamRef.current.getAudioTracks().forEach((track) => {
      track.enabled = enabled;
    });
  }, []);

  // 监听信令事件
  useEffect(() => {
    if (!socket || !bookingId) return;

    // 收到对方加入通知
    const handleUserJoined = async (data: { userId: string; role: string }) => {
      console.log('[WebRTC] 对方已加入:', data.role);
      // 如果自己是发起方且已有 PeerConnection，重新创建 Offer
      if (isInitiator && peerConnectionRef.current) {
        const offer = await peerConnectionRef.current.createOffer();
        await peerConnectionRef.current.setLocalDescription(offer);
        socket.emit('webrtc:offer', { bookingId, offer: peerConnectionRef.current.localDescription });
      }
    };

    // 收到 Offer
    const handleOffer = async (data: { userId: string; offer: RTCSessionDescriptionInit }) => {
      console.log('[WebRTC] 收到 Offer');
      const pc = peerConnectionRef.current || createPeerConnection();
      await pc.setRemoteDescription(new RTCSessionDescription(data.offer));
      const answer = await pc.createAnswer();
      await pc.setLocalDescription(answer);
      socket.emit('webrtc:answer', { bookingId, answer: pc.localDescription });
      console.log('[WebRTC] 已发送 Answer');
    };

    // 收到 Answer
    const handleAnswer = async (data: { userId: string; answer: RTCSessionDescriptionInit }) => {
      console.log('[WebRTC] 收到 Answer');
      const pc = peerConnectionRef.current;
      if (pc && pc.signalingState !== 'stable') {
        await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
      }
    };

    // 收到 ICE Candidate
    const handleIceCandidate = async (data: { userId: string; candidate: RTCIceCandidateInit }) => {
      const pc = peerConnectionRef.current;
      if (pc) {
        try {
          await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
        } catch (err) {
          console.warn('[WebRTC] 添加 ICE Candidate 失败:', err);
        }
      }
    };

    // 对方离开
    const handleUserLeft = () => {
      console.log('[WebRTC] 对方已离开');
      setRemoteStream(null);
      setIsConnected(false);
    };

    socket.on('webrtc:user-joined', handleUserJoined);
    socket.on('webrtc:offer', handleOffer);
    socket.on('webrtc:answer', handleAnswer);
    socket.on('webrtc:ice-candidate', handleIceCandidate);
    socket.on('webrtc:user-left', handleUserLeft);

    return () => {
      socket.off('webrtc:user-joined', handleUserJoined);
      socket.off('webrtc:offer', handleOffer);
      socket.off('webrtc:answer', handleAnswer);
      socket.off('webrtc:ice-candidate', handleIceCandidate);
      socket.off('webrtc:user-left', handleUserLeft);
    };
  }, [socket, bookingId, isInitiator, createPeerConnection]);

  // 清理
  useEffect(() => {
    return () => {
      peerConnectionRef.current?.close();
      localStreamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return {
    localStream,
    remoteStream,
    isConnected,
    isInitiator,
    error,
    startLocalStream,
    call,
    answerCall,
    hangUp,
    toggleCamera,
    toggleMic,
  };
}
