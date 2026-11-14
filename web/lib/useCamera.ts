'use client';
// Adapted from BooDarling lib/camera.ts (MIT). Added cancellation and visibility cleanup.
import {useCallback, useEffect, useRef, useState} from 'react';
export function useCamera() {
  const video = useRef<HTMLVideoElement>(null); const stream = useRef<MediaStream | null>(null); const generation = useRef(0);
  const [ready, setReady] = useState(false); const [pending, setPending] = useState(false); const [error, setError] = useState('');
  const stop = useCallback(() => { generation.current++; stream.current?.getTracks().forEach(t => t.stop()); stream.current = null; if (video.current) video.current.srcObject = null; setReady(false); setPending(false); }, []);
  useEffect(() => {
    const hide = () => { if (document.hidden) stop(); }; document.addEventListener('visibilitychange', hide);
    return () => {document.removeEventListener('visibilitychange', hide); stop();};
  }, [stop]);
  async function start() {
    stop(); const id = generation.current; setPending(true); setError('');
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw Error('unsupported');
      const s = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 960}}, audio: false});
      if (id !== generation.current) {s.getTracks().forEach(t => t.stop()); return;}
      stream.current = s;
      if (video.current) {video.current.srcObject = s; await video.current.play();}
      if (id !== generation.current) return;
      s.getVideoTracks().forEach(t => t.addEventListener('ended', () => {if (id === generation.current) {stop(); setError('Camera disconnected. You can continue with voice or text.');}}));
      setReady(true); setPending(false);
    } catch {if (id === generation.current) {stop(); setError('Camera unavailable or permission denied. Voice and text still work.');}}
  }
  return {video, ready, pending, error, start, stop};
}
