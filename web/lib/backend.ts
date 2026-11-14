import { isComparison, validateInput } from './contracts';
const headers = {'Cache-Control': 'no-store'};
export async function proxyBackend(endpoint: 'predict' | 'models' | 'health', request?: Request): Promise<Response> {
  let body: string | undefined;
  if (endpoint === 'predict') {
    if (!request) return Response.json({message: 'Request required.'}, {status: 400, headers});
    const origin = request.headers.get('origin');
    let sameOrigin = !origin;
    try { if (origin) sameOrigin = new URL(origin).host === (request.headers.get('host') || new URL(request.url).host); } catch { sameOrigin = false; }
    if (!sameOrigin) return Response.json({message: 'Use the website to submit this request.'}, {status: 403, headers});
    if (!request.headers.get('content-type')?.includes('application/json')) return Response.json({message: 'Expected JSON.'}, {status: 415, headers});
    // Bound the body even when Content-Length is absent or untrusted.
    const reader = request.body?.getReader(); let size = 0; const chunks: Uint8Array[] = [];
    if (reader) {
      while (true) {
        const {done, value} = await reader.read(); if (done) break;
        size += value.byteLength;
        if (size > 16000) { await reader.cancel(); return Response.json({message: 'Request too large.'}, {status: 413, headers}); }
        chunks.push(value);
      }
    }
    try {
      const raw = JSON.parse(Buffer.concat(chunks).toString('utf8'));
      if (!raw || typeof raw.text !== 'string' || !Array.isArray(raw.models) || raw.models.some((x: unknown) => typeof x !== 'string') || Object.keys(raw).some(k => !['text', 'models'].includes(k))) throw Error();
      const message = validateInput(raw.text, raw.models);
      if (message) return Response.json({message}, {status: 422, headers});
      body = JSON.stringify({text: raw.text.trim(), models: raw.models});
    } catch { return Response.json({message: 'Invalid prediction request.'}, {status: 422, headers}); }
  }
  try {
    const base = process.env.EMOTION_API_URL || 'http://127.0.0.1:8000';
    const upstream = await fetch(`${base.replace(/\/$/, '')}/${endpoint}`, {
      method: body ? 'POST' : 'GET', headers: body ? {'Content-Type': 'application/json'} : {}, body,
      cache: 'no-store', signal: AbortSignal.timeout(endpoint === 'predict' ? 45000 : 8000),
    });
    const data: unknown = await upstream.json();
    if (endpoint === 'predict' && (upstream.ok || upstream.status === 503) && !isComparison(data)) throw Error('Malformed upstream response');
    if (endpoint === 'predict' && !upstream.ok && upstream.status !== 503) return Response.json({message: 'The model service rejected this request. Check the text and selected models.'}, {status: upstream.status, headers});
    return Response.json(data, {status: upstream.status, headers});
  } catch (error) {
    const timeout = error instanceof Error && ['TimeoutError', 'AbortError'].includes(error.name);
    return Response.json({message: timeout ? 'The model service took too long. Your text is still here; retry when you are ready.' : 'Cannot reach the model service. Start the Python API and use Reconnect.'}, {status: timeout ? 504 : 502, headers});
  }
}
