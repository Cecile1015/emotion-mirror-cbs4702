'use client';
import {useEffect, useRef, useState} from 'react';
import {joinTranscript, Recognition, SpeechConstructor, speechError} from './speech';
export function useSpeech(text: string, update: (text: string) => void) {
  const [supported, setSupported] = useState<boolean | null>(null);
  const [phase, setPhase] = useState<'idle' | 'starting' | 'listening' | 'stopping'>('idle');
  const [error, setError] = useState(''); const [interim, setInterim] = useState('');
  const [seconds, setSeconds] = useState(0);
  const rec = useRef<Recognition | null>(null); const updateRef = useRef(update); updateRef.current = update;
  const timers = useRef<ReturnType<typeof setTimeout>[]>([]); const interval = useRef<ReturnType<typeof setInterval> | null>(null);
  const clear = () => { timers.current.forEach(clearTimeout); timers.current = []; if (interval.current) clearInterval(interval.current); interval.current = null; };
  useEffect(() => {
    const w = window as unknown as {SpeechRecognition?: SpeechConstructor; webkitSpeechRecognition?: SpeechConstructor};
    setSupported(!!(w.SpeechRecognition || w.webkitSpeechRecognition));
    const hide = () => { if (document.hidden) rec.current?.stop(); };
    document.addEventListener('visibilitychange', hide);
    return () => { document.removeEventListener('visibilitychange', hide); clear(); const r = rec.current; rec.current = null; if (r) {r.onstart = r.onend = r.onresult = r.onerror = null; r.abort();} };
  }, []);
  function stop() {
    if (!rec.current) return;
    const current = rec.current;
    setPhase('stopping'); current.stop();
    if (rec.current !== current) return;
    timers.current.push(setTimeout(() => { if (rec.current === current) {current.abort(); current.onend?.();} }, 2500));
  }
  function start() {
    if (rec.current) return;
    const w = window as unknown as {SpeechRecognition?: SpeechConstructor; webkitSpeechRecognition?: SpeechConstructor};
    const SR = w.SpeechRecognition || w.webkitSpeechRecognition;
    if (!SR) {setSupported(false); return;}
    const r = new SR(); rec.current = r; let final = text.trim(); let latest = ''; let hadSpeech = false; let failed = false;
    setError(''); setInterim(''); setPhase('starting'); setSeconds(0);
    r.lang = 'en-US'; r.continuous = true; r.interimResults = true; r.maxAlternatives = 1;
    r.onstart = () => {
      if (rec.current !== r) return; clear(); setPhase('listening');
      interval.current = setInterval(() => setSeconds(s => s + 1), 1000);
      timers.current.push(setTimeout(stop, 60000));
    };
    r.onresult = e => {
      if (rec.current !== r) return;
      const joined = joinTranscript(text, e.results); final = joined.final; latest = joined.interim;
      hadSpeech = true; updateRef.current(final); setInterim(latest);
    };
    r.onerror = e => { if (rec.current !== r) return; failed = true; if (e.error !== 'aborted') setError(speechError(e.error)); r.onend?.(); };
    r.onend = () => {
      if (rec.current !== r) return;
      clear(); rec.current = null; setPhase('idle'); setInterim('');
      if (latest) updateRef.current([final, latest].filter(Boolean).join(' '));
      if (!hadSpeech && !failed) setError(speechError('no-speech'));
      r.onstart = r.onend = r.onresult = r.onerror = null;
      // Ensure tracks held by the speech service are released after error/timeout.
      try { r.abort(); } catch { /* already ended */ }
    };
    timers.current.push(setTimeout(() => { if (rec.current === r) {failed = true; setError('The microphone did not start. Check browser permission, then try again.'); r.onend?.();} }, 12000));
    try {r.start();} catch {failed = true; setError('The microphone could not start. Try again, or type below.'); r.onend?.();}
  }
  return {supported, phase, error, interim, seconds, start, stop, active: phase !== 'idle'};
}
