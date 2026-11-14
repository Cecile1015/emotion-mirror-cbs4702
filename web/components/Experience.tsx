'use client';
import {useCallback, useEffect, useState} from 'react';
import {Model} from '@/lib/contracts';
import Mirror from './Mirror';
import Evidence from './Evidence';
import Icon from './Icon';
export default function Experience() {
  const [view, setView] = useState<'cover' | 'mirror' | 'ledger' | 'errors'>('cover'); const [seed, setSeed] = useState('');
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
  function navigate(next: typeof view) {setView(next); window.scrollTo({top: 0}); requestAnimationFrame(() => document.getElementById('main')?.focus());}
  return <><a className="skip-link" href="#main">Skip to content</a>{view !== 'cover' && <header className="site-header"><button className="brand" onClick={() => navigate('cover')} aria-label="Boo, Darlings home"><span className="brand-mark">✧</span><span>Boo, Darlings<small>EMOTION MIRROR · CBS4702</small></span></button><nav aria-label="Main navigation">{([['mirror','The Mirror'],['ledger','The Ledger'],['errors','Crypt of Errors']] as const).map(([id,label]) => <button key={id} aria-current={view === id ? 'page' : undefined} onClick={() => navigate(id)}>{label}</button>)}</nav><button className="connection" onClick={refresh} disabled={status === 'connecting'} title="Refresh model availability"><i className={`status-dot ${status === 'ready' ? 'on' : ''}`}/>{status === 'connecting' ? 'Connecting…' : status === 'offline' ? 'API offline · reconnect' : `${models.filter(m => m.available).length} models configured`}<span>↻</span></button></header>}
    <main id="main" tabIndex={-1} className={view === 'cover' ? 'cover-main' : undefined}>{view === 'cover' ? <section className="cover-gate" aria-labelledby="cover-title"><div className="cover-inner"><span className="cover-ornament" aria-hidden="true">❦</span><p className="cover-kicker">A CBS4702 EXPERIMENT</p><h1 id="cover-title" aria-label="Boo, Darlings: emotion mirror"><span aria-hidden="true" className="cover-glitch" data-text="Boo, Darlings:">Boo, Darlings:</span><em aria-hidden="true">emotion mirror</em></h1><p className="cover-note">Mirror, mirror on the wall…<br/>what do my words reveal?</p><div className="cover-actions"><button className="cover-enter" onClick={() => navigate('mirror')}>Open the emotion mirror <span aria-hidden="true">→</span></button><button className="cover-explore" onClick={() => navigate('ledger')}>Explore the evidence</button></div><p className="cover-caption">Speak. Reflect. Compare.<br/><span>Six text models · optional expression & voice</span></p><div className="cover-rule" aria-hidden="true"><span/>✧<span/></div><small className="cover-privacy">Your camera and microphone stay off until you choose to start.</small></div></section> : view === 'mirror' ? <Mirror saveDraft={setSeed} seed={seed} models={models} status={status} refresh={refresh}/> : <Evidence view={view} retryText={text => {setSeed(text); navigate('mirror');}}/>}</main>
    {view !== 'cover' && <footer className="site-footer"><div className="footer-brand"><Icon name="spark"/><span>For the curious, not the certain.</span></div><p>English emotion classification · six labels · real model predictions</p><div>CBS4702 · Research & reflection</div></footer>}
  </>;
}
