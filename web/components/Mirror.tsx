'use client';
import {useEffect, useRef, useState} from 'react';
import {Comparison, isComparison, Model, ModelId, MODEL_IDS, NAMES, NOTES, validateInput} from '@/lib/contracts';
import {useSpeech} from '@/lib/useSpeech';
import {useRecorder} from '@/lib/useRecorder';
import {useCamera} from '@/lib/useCamera';
import Icon from './Icon';
import Results from './Results';
import {useBonus} from '@/lib/useBonus';
import {BonusControls, BonusResults} from './Bonus';
export default function Mirror({seed, models, status, refresh, saveDraft}: {saveDraft: (text: string) => void; seed: string; models: Model[]; status: string; refresh: () => void}) {
  const [text, setText] = useState(seed); const [selected, setSelected] = useState<ModelId[]>(['tfidf_lr', 'bert']);
  const [textBusy, setBusy] = useState(false); const [error, setError] = useState(''); const [result, setResult] = useState<Comparison | null>(null); const [resultModels, setResultModels] = useState('');
  const camera = useCamera(); const bonus = useBonus(camera.video); const busy = textBusy || bonus.busy;
  const browserSpeech = useSpeech(text, setText); const recorder = useRecorder(text, setText, {maxSeconds:bonus.enabled?20:60,replace:bonus.enabled,onStart:bonus.begin,onRecorded:bonus.recorded,onCancel:bonus.reset}); const [voiceMode, setVoiceMode] = useState<'local' | 'browser'>('local'); const speech = voiceMode === 'local' ? recorder : browserSpeech; const request = useRef<AbortController | null>(null); const resultRef = useRef<HTMLDivElement>(null); const textarea = useRef<HTMLTextAreaElement>(null);
  useEffect(() => () => request.current?.abort(), []);
  useEffect(() => saveDraft(text), [text, saveDraft]);
  const available = (id: ModelId) => models.find(m => m.id === id)?.available ?? false;
  function toggle(id: ModelId) {setSelected(s => s.includes(id) ? s.filter(x => x !== id) : [...s, id]);}
  async function compare() {
    if (busy || speech.active) return;
    const invalid = validateInput(text, selected); if (invalid) {setError(invalid); textarea.current?.focus(); return;}
    setBusy(true); setError(''); void bonus.compare();
    const controller = new AbortController(); request.current = controller;
    const timer = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch('/api/predict', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, models: selected}), signal: controller.signal});
      const data: unknown = await response.json();
      if (!isComparison(data)) throw Error((data as {message?: string})?.message || 'The service returned an unexpected response. Your text is preserved.');
      if (!response.ok && response.status !== 503) throw Error('The comparison could not be completed. Please try again.');
      setResult(data); setResultModels(selected.join(','));
      setTimeout(() => resultRef.current?.scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth', block: 'start'}), 60);
    } catch (e) {
      if (request.current === controller) setError(controller.signal.aborted ? 'The comparison timed out. Your text is preserved. Try again when you are ready.' : e instanceof Error ? e.message : 'Connection lost. Your text is preserved.');
    } finally {clearTimeout(timer); if (request.current === controller) {request.current = null; setBusy(false);}}
  }
  const listeningText = speech.phase === 'starting' ? 'Waiting for microphone…' : speech.phase === 'stopping' ? (voiceMode === 'local' ? 'Transcribing your recording…' : 'Finishing your transcript…') : speech.phase === 'listening' ? `Listening · ${speech.seconds}s / ${bonus.enabled?20:60}s` : text ? 'Your words are ready to review' : 'The mirror awaits your words.';
  return <>
    <section className="intro"><div><p className="eyebrow">A VOICE, SIX PERSPECTIVES</p><h1>Mirror, mirror.<br/><em>How do I feel?</em></h1></div><div className="intro-copy"><span className="ornament">✧</span><p>Tell the mirror what’s on your mind.{' '}<br/>Discover how different models read the emotion in your words.</p><p className="small-caps">SPEAK. REFLECT. COMPARE.</p></div></section>
    <section className="mirror-workspace" aria-label="Speak and compare emotions">
      <div className="mirror-column">
        <div className={`frame ${speech.active ? 'listening-frame' : ''}`}>
          <span className="frame-flourish top" aria-hidden="true">❦</span><span className="frame-flourish bottom" aria-hidden="true">❦</span>
          <div className="glass">
            <video ref={camera.video} autoPlay playsInline muted className={camera.ready ? 'camera-video visible' : 'camera-video'} aria-label="Live mirrored camera preview" aria-hidden={!camera.ready}/>
            {!camera.ready && <div className="mirror-illustration" aria-hidden="true"><div className="halo"/><svg viewBox="0 0 280 320"><path d="M70 290c0-64 30-89 70-89s70 25 70 89"/><ellipse cx="140" cy="130" rx="54" ry="68"/><path d="M86 128c8-57 95-81 109 1M92 98c-12 37-14 73-30 91M189 98c14 37 16 73 29 91"/><path className="star-path" d="m225 53 5 13 13 5-13 5-5 13-5-13-13-5 13-5Z"/></svg><span className="mirror-whisper">a reflection of your words</span></div>}
            <div className="glass-top"><span className="glass-label"><i className={camera.ready ? 'status-dot on' : 'status-dot'}/>{camera.ready ? 'LIVE MIRROR' : 'YOUR REFLECTION'}</span><span className="glass-number">I · VI</span></div>
            <div className="glass-bottom"><div className={`voice-lines ${speech.phase === 'listening' ? 'moving' : ''}`} aria-hidden="true">{Array.from({length: 21}, (_, i) => <i key={i} style={{'--i': i} as React.CSSProperties}/>)}</div><p>{listeningText}</p><span>{bonus.enabled?'Selected expression and voice channels are experimental.':'Camera is a live preview; emotion comes from your words.'}</span></div>
          </div>
        </div>
        <div className="mirror-controls"><button className="text-button" onClick={camera.ready || camera.pending ? camera.stop : camera.start}><Icon name="camera"/>{camera.pending ? 'Cancel camera request' : camera.ready ? 'Turn camera off' : 'Turn mirror camera on'}</button><span>{bonus.enabled && bonus.channels.includes('face')?'Selected snapshots upload only when you compare':'Optional · stays in your browser'}</span></div>
        {camera.error && <p className="notice" role="status">{camera.error}</p>}
      </div>
      <div className="speech-panel">
        <div className="panel-topline"><span className="eyebrow">THE SPEAKING ROOM</span><span className="small-caps">ENGLISH · EN-US</span></div>
        <ol className="steps" aria-label="Workflow"><li className={!text || speech.active ? 'current' : ''}><span>01</span> Speak</li><li className={text && !speech.active && !result ? 'current' : ''}><span>02</span> Review</li><li className={result ? 'current' : ''}><span>03</span> Compare</li></ol>
        <h2>{speech.phase === 'listening' ? 'Go on. I’m listening.' : 'Say a little. Reveal a lot.'}</h2><p className="panel-description">A sentence about your day is a lovely place to start.</p>
        <BonusControls enabled={bonus.enabled} channels={bonus.channels} busy={busy || speech.active} ready={!!bonus.capture} error={bonus.error} toggle={()=>{bonus.toggle();setVoiceMode('local');}} choose={bonus.choose}/>
        <div className="voice-mode"><label htmlFor="voice-mode">Transcription</label><select id="voice-mode" value={voiceMode} disabled={speech.active || busy || bonus.enabled} onChange={e => setVoiceMode(e.target.value as 'local' | 'browser')}><option value="local">Whisper recording · recommended</option><option value="browser">Browser live captions</option></select></div>
        <button className={`speak-button ${speech.active ? 'recording' : ''}`} onClick={speech.active ? speech.stop : speech.start} disabled={busy || speech.supported === false || speech.phase === 'stopping' || (bonus.enabled && !bonus.channels.length)}><span className="mic-disc"><Icon name={speech.active ? 'stop' : 'mic'} size={25}/></span><span>{speech.phase === 'stopping' ? 'Transcribing…' : speech.active ? 'Stop & review' : 'Start speaking'}<small>{speech.active ? listeningText : voiceMode === 'local' ? 'Record, then review the transcript' : 'Your voice becomes an editable transcript'}</small></span><Icon name="arrow"/></button>
        <div className="speech-status" role="status" aria-live="polite">{speech.supported === false ? 'Voice is unavailable in this browser. Try Chrome or Edge, or type below.' : speech.error || (speech.phase === 'listening' ? (voiceMode === 'local' ? 'Recording in English. Stop to transcribe, then review your words.' : 'Speak in English. Stop when you are ready to review.') : 'Recording is sent for transcription after you stop. Review before emotion comparison.')}</div>
        {voiceMode === 'local' && recorder.active && <button className="text-button cancel-recording" onClick={recorder.cancel}>Cancel recording / transcription</button>}
        <div className="transcript-label"><label htmlFor="transcript">Your words <span>· editable transcript</span></label><button className="text-button" onClick={() => {setText(''); bonus.reset(); setError(''); textarea.current?.focus();}} disabled={busy || speech.active || !text}>Clear</button></div>
        <textarea id="transcript" ref={textarea} value={speech.interim ? `${text}${text ? ' ' : ''}${speech.interim}` : text} onChange={e => setText(e.target.value)} readOnly={speech.active || busy} rows={4} placeholder="I feel excited about our presentation…" aria-describedby="transcript-count"/>
        <div className="transcript-meta"><button className="text-button" disabled={speech.active || busy} onClick={() => {bonus.reset(); setText('i do not feel happy today'); textarea.current?.focus();}}>Try a sentence about negation ↗</button><span id="transcript-count" className={text.length > 2000 ? 'invalid' : ''}>{text.length.toLocaleString()} / 2,000</span></div>
        <fieldset disabled={busy || speech.active} className="model-fieldset"><legend>Choose your interpreters <span>{selected.length} selected</span></legend><div className="model-grid">{MODEL_IDS.map(id => <label key={id} className={`model-option ${selected.includes(id) ? 'selected' : ''} ${!available(id) ? 'unavailable' : ''}`}><input type="checkbox" checked={selected.includes(id)} disabled={!available(id) && !selected.includes(id)} onChange={() => toggle(id)}/><span>{NAMES[id]}<small>{!available(id) ? 'Unavailable · reconnect API' : NOTES[id]}</small></span>{id === 'llm' && <span className="paid-label">API</span>}</label>)}</div></fieldset>
        {selected.includes('llm') && <p className="remote-disclosure">DeepSeek sends this transcript to an external provider and uses the project’s API credit. Its confidence is reported as N/A.</p>}
        {error && <p className="notice error" role="alert">{error}</p>}
        <button className="compare-button" onClick={compare} disabled={busy || speech.active || !selected.length || status === 'connecting'}><Icon name="spark"/>{busy ? 'Reading your words…' : 'Compare emotions'}{busy ? <span className="spinner"/> : <Icon name="arrow"/>}</button>
        {busy && <p className="small-note" role="status">Running the selected models. DeepSeek may take longer; this request will not retry automatically.</p>}
        {status !== 'ready' && <button className="text-button reconnect" onClick={refresh} disabled={busy || status === 'connecting'}>{status === 'connecting' ? 'Connecting to model service…' : 'Reconnect model service'}</button>}
        <details className="privacy"><summary>A note on voice & privacy</summary><p>Local Whisper sends your recording to this project’s speech service after you stop; in local mode that service runs on this computer and keeps audio in memory only. If publicly hosted, audio goes to that server. The optional browser live-caption mode may send audio to the browser provider. Only the text you confirm is sent to the emotion classifiers. Camera frames stay in this browser unless you enable the expression channel in Mind-reading and compare. That opt-in sends up to three snapshots; the acoustic option sends the same recording for a separate voice model. The server processes media in memory without saving recordings or snapshots. Turning the bonus off or leaving the Mirror clears the retained media. Expression and voice scores are experimental clues, not hidden feelings.</p></details>
      </div>
    </section>
    <div ref={resultRef}>{result && <Results result={result} stale={result.text !== text.trim() || resultModels !== selected.join(',')}/>}<BonusResults result={bonus.result} busy={bonus.busy}/></div>
    {!result && <div className="quiet-note"><span>❧</span><p>There is more than one way to read a feeling.<br/><small>Compare the predictions. Question the confidence. Look closely at the language.</small></p><span>❧</span></div>}
  </>;
}
