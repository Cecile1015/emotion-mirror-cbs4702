import {test} from 'node:test';
import assert from 'node:assert/strict';
import {validateInput, isComparison} from '../lib/contracts';
import {joinTranscript, speechError} from '../lib/speech';
import {proxyBackend} from '../lib/backend';
test('input rejects blank, duplicate, unsupported models, long transcripts and control characters', () => {
  for (const [text, models] of [['   ', ['bert']], ['123', ['bert']], ['hello', []], ['hi',['bert','bert']], ['hi',['unknown']], ['a'.repeat(2001),['bert']], ['hello\u0000',['bert']]] as [string,string[]][]) assert.ok(validateInput(text,models));
  assert.equal(validateInput('I feel happy.', ['tfidf_lr','bert','llm']),null);
});
test('speech replaces revised interim results without duplicating final segments or the starting text', () => {
  const results = [{isFinal:true,0:{transcript:'I feel'}},{isFinal:false,0:{transcript:'happy'}}];
  assert.deepEqual(joinTranscript('Today.',results),{final:'Today. I feel',interim:'happy'});
  results[1] = {isFinal:true,0:{transcript:'excited'}};
  assert.deepEqual(joinTranscript('Today.',results),{final:'Today. I feel excited',interim:''});
  assert.deepEqual(joinTranscript('',[]),{final:'',interim:''});
});
test('speech failures explain permissions, silence and network with text fallback', () => {
  for(const code of ['not-allowed','audio-capture','no-speech','network','service-not-allowed','language-not-supported']) assert.match(speechError(code),/typ/i);
});
test('a DeepSeek label has no invented confidence and malformed results are rejected', () => {
  const data = {text:'hello', results:[{model_id:'llm',label:'joy',confidence:null,probabilities:null,warnings:[]}],errors:[],all_agree:null};
  assert.equal(isComparison(data),true);
  assert.equal(isComparison({...data,results:[{...data.results[0],confidence:1}]}),false);
  assert.equal(isComparison({...data,results:[{...data.results[0],label:'neutral'}]}),false);
});
test('proxy bounds requests and refuses foreign-origin POST without upstream calls', async () => {
  const send = (body: string, origin='http://localhost:3000') => new Request('http://localhost:3000/api/predict',{method:'POST',headers:{'Content-Type':'application/json',Origin:origin},body});
  assert.equal((await proxyBackend('predict',send('x'.repeat(16001)))).status,413);
  assert.equal((await proxyBackend('predict',send('not json'))).status,422);
  assert.equal((await proxyBackend('predict',send('{}','https://other.example'))).status,403);
});
test('proxy forwards the same text/models, preserves partial failures and sanitizes network failures', async () => {
  const original = globalThis.fetch; const env = process.env.EMOTION_API_URL;
  process.env.EMOTION_API_URL = 'http://backend.example';
  const payload = {text:'hello',models:['bert','llm']};
  const result = {text:'hello',results:[],errors:[{model_id:'llm',code:'prediction_failed',message:'Provider unavailable'}],all_agree:null};
  try {
    let count=0;
    globalThis.fetch = async (url,options) => {count++;assert.equal(url,'http://backend.example/predict');assert.deepEqual(JSON.parse(options!.body as string),payload);return Response.json(result,{status:503});};
    const response = await proxyBackend('predict',new Request('http://localhost/api/predict',{method:'POST',headers:{'Content-Type':'application/json',Host:'127.0.0.1:3000',Origin:'http://127.0.0.1:3000'},body:JSON.stringify(payload)}));
    assert.equal(response.status,503);assert.deepEqual(await response.json(),result);assert.equal(count,1);
    globalThis.fetch = async () => {throw Error('sensitive backend details');};
    const failure = await proxyBackend('models');assert.equal(failure.status,502);assert.doesNotMatch(await failure.text(),/sensitive/);
  } finally {globalThis.fetch=original;if(env === undefined) delete process.env.EMOTION_API_URL; else process.env.EMOTION_API_URL=env;}
});
