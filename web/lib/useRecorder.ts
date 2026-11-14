'use client';
import {useEffect, useRef, useState} from 'react';
export function useRecorder(text: string, update: (s: string) => void, options: {maxSeconds?:number; replace?:boolean; onStart?:()=>void; onRecorded?:(blob:Blob)=>void; onCancel?:()=>void} = {}) {
  const [supported,setSupported]=useState<boolean|null>(null);const [phase,setPhase]=useState<'idle'|'starting'|'listening'|'stopping'>('idle');const [error,setError]=useState('');const [seconds,setSeconds]=useState(0);
  const rec=useRef<MediaRecorder|null>(null);const stream=useRef<MediaStream|null>(null);const request=useRef<AbortController|null>(null);const generation=useRef(0);const timers=useRef<ReturnType<typeof setTimeout>[]>([]);const interval=useRef<ReturnType<typeof setInterval>|null>(null);
  const updateRef=useRef(update);updateRef.current=update;const optionsRef=useRef(options);optionsRef.current=options;
  function clearTimers(){timers.current.forEach(clearTimeout);timers.current=[];if(interval.current)clearInterval(interval.current);interval.current=null;}
  function release(){stream.current?.getTracks().forEach(t=>t.stop());stream.current=null;clearTimers();}
  function cancel(){optionsRef.current.onCancel?.();generation.current++;request.current?.abort();request.current=null;const r=rec.current;rec.current=null;if(r){r.onstop=r.ondataavailable=r.onerror=null;if(r.state!=='inactive')r.stop();}release();setPhase('idle');}
  function stop(){const r=rec.current;if(!r){cancel();return;}if(r.state!=='inactive'){setPhase('stopping');r.stop();}}
  useEffect(()=>{setSupported(!!navigator.mediaDevices?.getUserMedia && typeof MediaRecorder!=='undefined');const hide=()=>{if(document.hidden)cancel();};document.addEventListener('visibilitychange',hide);return()=>{document.removeEventListener('visibilitychange',hide);cancel();};},[]);
  async function start(){
    if(rec.current || phase!=='idle')return;
    const id=++generation.current;setError('');setPhase('starting');setSeconds(0);
    timers.current.push(setTimeout(()=>{if(generation.current===id){cancel();setError('Microphone permission is still pending. Allow access, then try again.');}},15000));
    try{
      const s=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true},video:false});
      if(generation.current!==id){s.getTracks().forEach(t=>t.stop());return;}
      clearTimers();stream.current=s;
      const mime=['audio/webm;codecs=opus','audio/mp4','audio/ogg;codecs=opus'].find(t=>MediaRecorder.isTypeSupported(t));
      const r=new MediaRecorder(s,mime?{mimeType:mime}:undefined);rec.current=r;const chunks:Blob[]=[];let bytes=0;
      r.ondataavailable=e=>{if(e.data.size){bytes+=e.data.size;if(bytes>8*1024*1024){cancel();setError('Recording is too large. Try a shorter sentence.');return;}chunks.push(e.data);}};
      r.onerror=()=>{if(generation.current===id){cancel();setError('Recording failed. Your existing text is preserved. Check the microphone and try again.');}};
      r.onstop=async()=>{
        release();rec.current=null;if(generation.current!==id)return;setPhase('stopping');
        const c=new AbortController();request.current=c;const timeout=setTimeout(()=>c.abort(),90000);
        try{
          const blob=new Blob(chunks,{type:r.mimeType||'audio/webm'});
          if(!blob.size)throw Error('The recording was empty. Please try again.');
          optionsRef.current.onRecorded?.(blob);
          const response=await fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':blob.type},body:blob,signal:c.signal});
          const data=await response.json();if(!response.ok || typeof data.text!=='string')throw Error(data.message||data.detail||'Transcription failed. Please try again.');
          if(generation.current===id)updateRef.current(optionsRef.current.replace?data.text.trim():[text.trim(),data.text.trim()].filter(Boolean).join(' '));
        }catch(e){if(generation.current===id)setError(c.signal.aborted?'Transcription took too long. Please try a shorter recording.':e instanceof Error?e.message:'The speech service could not be reached.');}
        finally{clearTimeout(timeout);if(generation.current===id){request.current=null;setPhase('idle');}}
      };
      r.start(1000);optionsRef.current.onStart?.();setPhase('listening');interval.current=setInterval(()=>setSeconds(n=>n+1),1000);timers.current.push(setTimeout(stop,(optionsRef.current.maxSeconds||60)*1000));
    }catch(e){if(generation.current===id){cancel();setError(e instanceof DOMException && e.name==='NotAllowedError'?'Microphone access was denied. Allow it in browser settings, then try again, or type below.':'No working microphone was found. Check your input device, or type below.');}}
  }
  return {supported,phase,error,interim:'',seconds,start,stop,cancel,active:phase!=='idle'};
}
