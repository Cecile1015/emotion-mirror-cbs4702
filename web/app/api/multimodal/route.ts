export const runtime='nodejs';
export const dynamic='force-dynamic';
export async function POST(request:Request){
  const headers={'Cache-Control':'no-store'};
  try{
    const origin=request.headers.get('origin');
    if(origin && new URL(origin).host!==(request.headers.get('host')||new URL(request.url).host))return Response.json({message:'Use the website to compare media.'},{status:403,headers});
    if(!request.headers.get('content-type')?.startsWith('application/json'))return Response.json({message:'Expected JSON.'},{status:415,headers});
    const chunks:Uint8Array[]=[];let size=0;const reader=request.body?.getReader();
    if(reader)while(true){const {done,value}=await reader.read();if(done)break;size+=value.byteLength;if(size>8*1024*1024){await reader.cancel();return Response.json({message:'Upload exceeds 8 MB.'},{status:413,headers});}chunks.push(value);}
    const upstream=await fetch(`${(process.env.EMOTION_SPEECH_URL||'http://127.0.0.1:8001').replace(/\/$/,'')}/multimodal`,{method:'POST',headers:{'Content-Type':'application/json'},body:new Uint8Array(Buffer.concat(chunks)).buffer,signal:AbortSignal.timeout(85000),cache:'no-store'});
    return Response.json(await upstream.json(),{status:upstream.status,headers});
  }catch{return Response.json({message:'Bonus service unavailable or timed out. Text comparison still works.'},{status:503,headers});}
}
