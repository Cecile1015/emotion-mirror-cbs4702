import {Comparison, LABELS, NAMES} from '@/lib/contracts';
export default function Results({result, stale}: {result: Comparison; stale: boolean}) {
  return <section className="results" aria-labelledby="results-heading">
    <div className="section-heading"><div><p className="eyebrow">THE REFLECTION</p><h2 id="results-heading">One sentence. Different readings.</h2></div>
      <span className="agreement">{result.all_agree === null ? (result.errors.length ? 'Comparison incomplete' : 'One model selected') : result.all_agree ? 'Selected models agree' : 'The models disagree'}</span></div>
    <blockquote className="submitted-text">“{result.text}”</blockquote>
    {stale && <p className="notice">Your input or model selection has changed. These are the previous results; compare again to update them.</p>}
    <div className="result-grid">{result.results.map(r => <article className="result-card" key={r.model_id}>
      <div className="card-heading"><span>{NAMES[r.model_id]}</span><small>{r.provider === 'deepseek' ? 'REMOTE' : 'LOCAL'}</small></div>
      <div className="verdict"><span className={`emotion ${r.label}`}>{r.label}</span><span className="confidence">{r.confidence === null ? 'N/A' : `${(r.confidence * 100).toFixed(1)}%`}<small>class confidence</small></span></div>
      {r.probabilities ? <div className="probabilities">{LABELS.map(l => <div className="probability" key={l}><span>{l}</span><div className="bar-track"><span className={l} style={{width: `${r.probabilities![l] * 100}%`}}/></div><span>{(r.probabilities![l] * 100).toFixed(1)}%</span></div>)}</div> : <div className="remote-note"><span className="ornament">✧</span><p>A few-shot reading from DeepSeek.</p><small>This configuration provides a label, without a validated six-class probability distribution.</small></div>}
      <div className="result-meta">{r.provider === 'deepseek' ? <>{r.served_model} · {r.n_shots} demonstrations<br/>{r.usage?.prompt_tokens} input / {r.usage?.completion_tokens} output tokens</> : <>{r.calibrated ? `Validation-calibrated · T ${r.temperature?.toFixed(3)}` : 'Uncalibrated baseline'}<br/>{Math.round(r.elapsed_ms)} ms inference</>}</div>
      {r.truncated && <p className="notice">This model truncated the input. Models may have seen different spans of text.</p>}
    </article>)}
    {result.errors.map(e => <article className="result-card error-card" key={e.model_id}><div className="card-heading">{NAMES[e.model_id]}</div><h3>No prediction</h3><p>{e.message}</p><small>Other successful results are preserved.</small></article>)}</div>
    <p className="fine-print">Confidence describes a model’s distribution over six labels. It is not emotional intensity or a measure of your psychological state. A disagreement is an invitation to examine the language.</p>
  </section>;
}
