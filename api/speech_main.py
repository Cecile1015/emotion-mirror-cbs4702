"""Local English ASR sidecar. No audio is written to disk or sent to an external service.

Run: python -m uvicorn api.speech_main:app --host 127.0.0.1 --port 8001
"""
from io import BytesIO
from pathlib import Path
import threading

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

MODEL_DIR = Path(__file__).resolve().parents[1] / 'models/speech/tiny.en'
MAX_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 62  # recording stops at 60 seconds; allow encoder padding
TYPES = {'audio/webm', 'audio/mp4', 'audio/ogg', 'audio/wav', 'audio/x-wav', 'audio/aiff', 'audio/x-aiff'}
app = FastAPI(title='Emotion Mirror Local Speech', version='1.0.0')
_lock = threading.Lock()
_model = None


def decode_bounded(data):
    import av
    chunks, samples = [], 0
    try:
        with av.open(BytesIO(data)) as container:
            if not container.streams.audio:
                raise ValueError('No audio stream')
            resampler = av.AudioResampler(format='fltp', layout='mono', rate=16000)
            for frame in container.decode(audio=0):
                for converted in resampler.resample(frame):
                    samples += converted.samples
                    if samples > MAX_SECONDS * 16000:
                        raise HTTPException(422, 'Keep recordings under 60 seconds.')
                    chunks.append(converted.to_ndarray().flatten())
            for converted in resampler.resample(None):
                samples += converted.samples
                if samples > MAX_SECONDS * 16000: raise HTTPException(422, 'Keep recordings under 60 seconds.')
                chunks.append(converted.to_ndarray().flatten())
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, 'The recording could not be decoded. Please record again.')
    if not chunks or samples < 1600:
        raise HTTPException(422, 'The recording is too short. Please speak a complete sentence.')
    return np.concatenate(chunks).astype(np.float32)


def transcribe_bytes(data):
    global _model
    if not (MODEL_DIR / 'model.bin').is_file():
        raise HTTPException(503, 'Prepare the local speech model with scripts/prepare_speech.py.')
    if not _lock.acquire(blocking=False):
        raise HTTPException(429, 'The speech service is busy. Please try again shortly.')
    try:
        audio = decode_bounded(data)
        if np.sqrt(np.mean(audio ** 2)) < 0.0003:
            raise HTTPException(422, 'No speech was detected. Check the microphone and try again.')
        if _model is None:
            from faster_whisper import WhisperModel
            _model = WhisperModel(str(MODEL_DIR), device='cpu', compute_type='int8', cpu_threads=4, local_files_only=True)
        segments, info = _model.transcribe(audio, language='en', beam_size=3, vad_filter=True,
                                         condition_on_previous_text=False, temperature=0)
        text = ' '.join(s.text.strip() for s in segments).strip()
        if not text:
            raise HTTPException(422, 'No speech was detected. Please try speaking closer to the microphone.')
        return {'text': text, 'language': 'en', 'model': 'whisper-tiny.en', 'duration_seconds': round(len(audio) / 16000, 2),
                'provider': 'local', 'warning': 'Review and correct the transcript before emotion comparison.'}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(503, 'Local transcription failed. Check the speech service dependencies and try again.')
    finally:
        _lock.release()


@app.get('/health')
def health():
    return {'configured': (MODEL_DIR / 'model.bin').is_file(), 'loaded': _model is not None,
            'model': 'whisper-tiny.en', 'provider': 'local', 'max_seconds': 60}


@app.post('/transcribe')
async def transcribe(request: Request):
    mime = request.headers.get('content-type', '').split(';')[0].lower()
    if mime not in TYPES: raise HTTPException(415, 'Unsupported recording format.')
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES: raise HTTPException(413, 'Recording exceeds the 8 MB limit.')
        data.extend(chunk)
    if not data: raise HTTPException(422, 'The recording is empty.')
    return await run_in_threadpool(transcribe_bytes, bytes(data))

# Optional bonus shares the local media sidecar, not the text-classifier contract.
from .multimodal import MAX_BODY, MultimodalRequest, service as multimodal_service
from pydantic import ValidationError
import json

@app.get('/multimodal/health')
def multimodal_health():
    return multimodal_service.health()

@app.post('/multimodal')
async def multimodal(request: Request):
    if request.headers.get('content-type', '').split(';')[0] != 'application/json':
        raise HTTPException(415, 'Expected JSON.')
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BODY: raise HTTPException(413, 'Bonus upload exceeds 8 MB.')
        data.extend(chunk)
    try: payload = MultimodalRequest.model_validate(json.loads(data))
    except (ValueError, ValidationError): raise HTTPException(422, 'Invalid bonus request. Choose channels and record again.')
    return await run_in_threadpool(multimodal_service.predict, payload)
