import test from 'node:test';
import assert from 'node:assert/strict';
import {POST} from '../app/api/multimodal/route';
test('bonus proxy rejects foreign origin and wrong media type before inference',async()=>{
  assert.equal((await POST(new Request('http://localhost/api/multimodal',{method:'POST',headers:{origin:'https://elsewhere.invalid','Content-Type':'application/json'},body:'{}'}))).status,403);
  assert.equal((await POST(new Request('http://localhost/api/multimodal',{method:'POST',body:'{}'}))).status,415);
});
test('bonus proxy bounds a streamed body without Content-Length',async()=>{
  const r=await POST(new Request('http://localhost/api/multimodal',{method:'POST',headers:{'Content-Type':'application/json'},body:'x'.repeat(8*1024*1024+1)}));
  assert.equal(r.status,413);
});
test('bonus proxy preserves independent channel errors',async()=>{
  const original=globalThis.fetch;
  const result={capture_id:'test',results:[{channel:'acoustic',label:'neu'}],errors:[{channel:'face',message:'No face'}]};
  try{
    globalThis.fetch=async(_url,init)=>{assert.equal(JSON.parse(Buffer.from(init!.body as ArrayBuffer).toString()).capture_id,'test');return Response.json(result);};
    const r=await POST(new Request('http://localhost/api/multimodal',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({capture_id:'test'})}));
    assert.deepEqual(await r.json(),result);
  }finally{globalThis.fetch=original;}
});
