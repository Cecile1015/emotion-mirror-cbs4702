'use client';
import {RefObject, useEffect, useRef, useState} from 'react';
export type Channel = 'face' | 'acoustic';
type Frame = {image: string; offset_ms: number};
export type BonusResult = {capture_id: string; note: string; results: {channel: Channel; model: string; label: string; confidence: number; probabilities: Record<string,number>; elapsed_ms: number; warnings: string[]; frames_used?: number; features?: {median_f0_hz: number|null; rms_dbfs: number; voiced_fraction: number}; duration_seconds?: number}[]; errors: {channel: Channel; message: string}[]};
type Capture = {id: string; audio: Blob; frames: Frame[]; channels: Channel[]};
async function encode(blob: Blob) {const bytes=new Uint8Array(await blob.arrayBuffer());let value='';for(let i=0;i<bytes.length;i+=8192)value+=String.fromCharCode(...bytes.subarray(i,i+8192));return btoa(value);}
export function useBonus(video: RefObject<HTMLVideoElement|null>) {
  const [enabled,setEnabled]=useState(false); const [channels,setChannels]=useState<Channel[]>(['face','acoustic']);
  const [capture,setCapture]=useState<Capture|null>(null); const [result,setResult]=useState<BonusResult|null>(null); const [busy,setBusy]=useState(false);const [error,setError]=useState('');
  const frames=useRef<Frame[]>([]); const started=useRef(0);const active=useRef(false);const selected=useRef<Channel[]>([]);
  const interval=useRef<ReturnType<typeof setInterval>|null>(null);const request=useRef<AbortController|null>(null);
  function clearTimer(){if(interval.current)clearInterval(interval.current);interval.current=null;}
  function reset(){active.current=false;clearTimer();frames.current=[];setCapture(null);request.current?.abort();request.current=null;setBusy(false);setResult(null);setError('');}
  useEffect(()=>()=>{clearTimer();request.current?.abort();},[]);
  function snapshot(){
    if(!active.current || !selected.current.includes('face'))return;
    const v=video.current;if(!v || v.readyState<2 || !v.videoWidth || !v.srcObject)return;
    const canvas=document.createElement('canvas');canvas.width=640;canvas.height=Math.round(v.videoHeight*640/v.videoWidth);
    if(canvas.height>1600)return;
    canvas.getContext('2d')?.drawImage(v,0,0,canvas.width,canvas.height);
    const frame={image:canvas.toDataURL('image/jpeg',.8).split(',')[1],offset_ms:Math.min(21000,Math.round(performance.now()-started.current))};
    frames.current=[...frames.current,frame].slice(-3);
  }
  function begin(){reset();if(!enabled)return;selected.current=[...channels];started.current=performance.now();active.current=true;snapshot();interval.current=setInterval(snapshot,4000);}
  function recorded(audio: Blob){if(!active.current)return;snapshot();clearTimer();active.current=false;setCapture({id:crypto.randomUUID(),audio,frames:[...frames.current],channels:[...selected.current]});frames.current=[];}
  function toggle(){reset();setEnabled(v=>!v);}
  function choose(channel:Channel){reset();setChannels(list=>list.includes(channel)?list.filter(x=>x!==channel):[...list,channel]);}
  async function compare(){
    if(!enabled)return;
    if(!capture){setError('Record a new utterance with Mind-reading enabled before comparing these channels.');return;}
    const controller=new AbortController();request.current=controller;setBusy(true);setError('');
    const timer=setTimeout(()=>controller.abort(),90000);
    try{
      const response=await fetch('/api/multimodal',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({capture_id:capture.id,channels:capture.channels,frames:capture.channels.includes('face')?capture.frames:[],audio:capture.channels.includes('acoustic')?await encode(capture.audio):null})});
      const data=await response.json();if(!response.ok || data.capture_id!==capture.id || !Array.isArray(data.results)||!Array.isArray(data.errors))throw Error(data.message||data.detail||'Bonus inference failed. Text results are preserved.');
      if(request.current===controller)setResult(data);
    }catch(e){if(request.current===controller)setError(controller.signal.aborted?'Bonus inference timed out. Text results are preserved.':e instanceof Error?e.message:'Bonus service unavailable.');}
    finally{clearTimeout(timer);if(request.current===controller){request.current=null;setBusy(false);}}
  }
  return {enabled,channels,capture,result,busy,error,toggle,choose,begin,recorded,reset,compare};
}
