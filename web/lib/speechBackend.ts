const headers={'Cache-Control':'no-store'};
export async function speechBackend(request?:Request){
  const base=(process.env.EMOTION_SPEECH_URL||'http://127.0.0.1:8001').replace(/\/$/,'');
  try{
    let body:Buffer|undefined;
    if(request){
      const origin=request.headers.get('origin');
      if(origin && new URL(origin).host !== (request.headers.get('host')||new URL(request.url).host))return Response.json({message:'Use the website to record audio.'},{status:403,headers});
      const reader=request.body?.getReader();const chunks:Uint8Array[]=[];let length=0;
      if(reader)while(true){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>8*1024*1024){await reader.cancel();return Response.json({message:'Recording exceeds 8 MB.'},{status:413,headers});}chunks.push(value);}
      body=Buffer.concat(chunks);
    }
    const response=await fetch(`${base}/${request?'transcribe':'health'}`,{method:request?'POST':'GET',headers:request?{'Content-Type':request.headers.get('content-type')||'audio/webm'}:{},body:body?new Uint8Array(body).buffer:undefined,signal:AbortSignal.timeout(request?80000:5000),cache:'no-store'});
    return Response.json(await response.json(),{status:response.status,headers});
  }catch{return Response.json({message:'Local speech service is unavailable or timed out. Start the website launcher, then try again; typing still works.'},{status:503,headers});}
}
