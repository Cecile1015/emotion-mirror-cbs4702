'use client';
import {useCallback, useEffect, useState} from 'react';
import {Model} from '@/lib/contracts';
import Mirror from './Mirror';
import Evidence from './Evidence';
import Icon from './Icon';
export default function Experience() {
  const [view, setView] = useState<'mirror' | 'ledger' | 'errors'>('mirror'); const [seed, setSeed] = useState('');
  const [models, setModels] = useState<Model[]>([]); const [status, setStatus] = useState('connecting');
  const [connectionVersion, setConnectionVersion] = useState(0);
  const refresh = useCallback(() => setConnectionVersion(v => v + 1), []);
  useEffect(() => {
    let active = true; const controller = new AbortController(); const timer = setTimeout(() => controller.abort(), 10000);
    setStatus('connecting');
    fetch('/api/models', {signal: controller.signal, cache: 'no-store'}).then(async r => {
      const d = await r.json(); if (!r.ok || !Array.isArray(d.models)) throw Error();
      if (!active) return; setModels(d.models); setStatus(d.models.filter((m: Model) => m.id !== 'llm').every((m: Model) => m.available) ? 'ready' : 'partial');
    }).catch(() => {if (active) {setModels([]); setStatus('offline');}}).finally(() => clearTimeout(timer));
    return () => {active = false; clearTimeout(timer); controller.abort();};
  }, [connectionVersion]);
  function navigate(next: typeof view) {setView(next); window.scrollTo({top: 0});}
  return <><a className="skip-link" href="#main">Skip to content</a><header className="site-header"><button className="brand" onClick={() => navigate('mirror')} aria-label="Emotion Mirror home"><span className="brand-mark">✧</span><span>Emotion Mirror<small>A CBS4702 EXPERIMENT</small></span></button><nav aria-label="Main navigation">{([['mirror','The Mirror'],['ledger','The Ledger'],['errors','Crypt of Errors']] as const).map(([id,label]) => <button key={id} aria-current={view === id ? 'page' : undefined} onClick={() => navigate(id)}>{label}</button>)}</nav><button className="connection" onClick={refresh} disabled={status === 'connecting'} title="Refresh model availability"><i className={`status-dot ${status === 'ready' ? 'on' : ''}`}/>{status === 'connecting' ? 'Connecting…' : status === 'offline' ? 'API offline · reconnect' : `${models.filter(m => m.available).length} models configured`}<span>↻</span></button></header>
    <main id="main">{view === 'mirror' ? <Mirror saveDraft={setSeed} seed={seed} models={models} status={status} refresh={refresh}/> : <Evidence view={view} retryText={text => {setSeed(text); navigate('mirror');}}/>}</main>
    <footer className="site-footer"><div className="footer-brand"><Icon name="spark"/><span>For the curious, not the certain.</span></div><p>English emotion classification · six labels · real model predictions</p><div><span>Design adapted from </span><a href="https://github.com/Cecile1015/BooDarling_CBS4702" target="_blank" rel="noreferrer">Boo, Darling! ↗</a><span> · MIT · Kevin Dai</span></div></footer>
  </>;
}
