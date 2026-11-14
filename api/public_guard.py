"""Single-process public-demo limits; counters store no text, media or client IPs."""
import os
import threading
import time
from collections import deque
from starlette.responses import JSONResponse

class PublicGuard:
    def __init__(self, app):
        self.app=app; self.lock=threading.Lock();self.events=deque();self.media=deque();self.running=0
    async def __call__(self, scope, receive, send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        path=scope['path']; method=scope['method']
        protected=method=='POST' and path in ('/api/predict','/api/transcribe','/api/multimodal')
        if not protected:return await self.app(scope,receive,send)
        headers=dict(scope.get('headers',[]));origin=headers.get(b'origin',b'').decode()
        from urllib.parse import urlparse
        if origin and urlparse(origin).netloc!=headers.get(b'host',b'').decode():
            return await JSONResponse({'message':'Use this website to submit requests.'},status_code=403)(scope,receive,send)
        maximum=16000 if path=='/api/predict' else 8*1024*1024
        body=bytearray()
        while True:
            message=await receive()
            if message['type']=='http.disconnect':return
            body.extend(message.get('body',b''))
            if len(body)>maximum:return await JSONResponse({'message':'Request exceeds the size limit.'},status_code=413)(scope,receive,send)
            if not message.get('more_body'):break
        now=time.monotonic();is_media=path!='/api/predict'
        with self.lock:
            while self.events and self.events[0]<now-60:self.events.popleft()
            while self.media and self.media[0]<now-60:self.media.popleft()
            rejected=self.running>=2 or len(self.events)>=20 or (is_media and len(self.media)>=6)
            if not rejected:
                self.events.append(now)
                if is_media:self.media.append(now)
                self.running+=1
        if rejected:return await JSONResponse({'message':'This shared demo is busy or its minute limit was reached. Please try again shortly.'},status_code=429,headers={'Retry-After':'30'})(scope,receive,send)
        consumed=False
        async def replay():
            nonlocal consumed
            if consumed:return await receive()
            consumed=True;return {'type':'http.request','body':bytes(body),'more_body':False}
        async def add_headers(message):
            if message['type']=='http.response.start':
                message.setdefault('headers',[]).extend([(b'cache-control',b'no-store'),(b'x-content-type-options',b'nosniff')])
            await send(message)
        try:return await self.app(scope,replay,add_headers)
        finally:
            with self.lock:self.running-=1

class RemoteBudget:
    """Reserve two calls per comparison (initial + optional repair). No credit balance promise."""
    def __init__(self):self.lock=threading.Lock();self.used=0
    def reserve(self):
        # Cloud default is set explicitly in the Space entry point. Local tests retain original behaviour.
        limit=int(os.getenv('EMOTION_DEEPSEEK_SESSION_CALL_LIMIT','0'))
        if limit<=0:return
        with self.lock:
            if self.used+2>limit:raise RuntimeError('Shared demo DeepSeek call allowance exhausted.')
            self.used+=2
remote_budget=RemoteBudget()
