"""Opt-in, independent expression and acoustic inference; media remain in memory."""
from __future__ import annotations
import base64
import binascii
import json
import threading
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Literal
import numpy as np
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / 'models/multimodal'
MAX_BODY = 8 * 1024 * 1024
FACE_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
AUDIO_LABELS = ['neu', 'hap', 'ang', 'sad']

class Frame(BaseModel):
    model_config = ConfigDict(extra='forbid')
    offset_ms: int = Field(ge=0, le=21000)
    image: str = Field(min_length=4, max_length=700000)

class MultimodalRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    capture_id: str = Field(min_length=1, max_length=80, pattern=r'^[a-zA-Z0-9_-]+$')
    channels: list[Literal['face', 'acoustic']] = Field(min_length=1, max_length=2)
    frames: list[Frame] = Field(default_factory=list, max_length=3)
    audio: str | None = Field(default=None, max_length=5600000)
    @model_validator(mode='after')
    def independent_consent(self):
        if len(set(self.channels)) != len(self.channels): raise ValueError('Choose unique channels.')
        if self.frames and 'face' not in self.channels: raise ValueError('Unselected image upload.')
        if self.audio and 'acoustic' not in self.channels: raise ValueError('Unselected audio upload.')
        return self

def decode64(value, maximum):
    try: data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error): raise ValueError('Invalid media encoding. Record again.')
    if not data or len(data) > maximum: raise ValueError('Media exceeds the allowed size.')
    return data

def image_from_bytes(data):
    from PIL import Image, ImageOps
    try:
        with Image.open(BytesIO(data)) as original:
            if original.format not in ('JPEG', 'PNG') or max(original.size) > 1600 or min(original.size) < 48:
                raise ValueError('Use a JPEG/PNG frame between 48 and 1600 pixels.')
            original.load()
            return ImageOps.exif_transpose(original).convert('RGB')
    except ValueError: raise
    except Exception: raise ValueError('Image could not be decoded. Record again.')

def acoustic_features(audio, sample_rate=16000):
    """Descriptive autocorrelation F0; not a rule mapping pitch to emotion."""
    rms = float(np.sqrt(np.mean(audio ** 2)))
    pitches = []
    total = 0
    for start in range(0, len(audio) - 640 + 1, 320):
        frame = audio[start:start+640].astype(np.float64)
        total += 1
        if np.sqrt(np.mean(frame ** 2)) < max(.002, rms * .2): continue
        frame -= frame.mean()
        corr = np.correlate(frame, frame, mode='full')[639:]
        if corr[0] < 1e-10: continue
        low, high = int(sample_rate / 400), int(sample_rate / 70)
        lag = low + int(np.argmax(corr[low:high+1]))
        if corr[lag] / corr[0] >= .6: pitches.append(sample_rate / lag)
    return {'rms_dbfs': round(20 * np.log10(max(rms, 1e-9)), 2),
            'median_f0_hz': round(float(np.median(pitches)), 1) if len(pitches) >= 10 else None,
            'voiced_fraction': round(len(pitches) / max(total, 1), 3),
            'clipping_fraction': round(float(np.mean(np.abs(audio) >= .995)), 4)}

class MultimodalService:
    def __init__(self):
        self.lock = threading.Lock()
        self.face_model = self.face_processor = self.detector = None
        self.audio_model = self.audio_processor = None

    def health(self):
        return {'experimental': True, 'max_seconds': 20, 'channels': {
            'face': {'model': 'trpakov/vit-face-expression', 'labels': FACE_LABELS,
                     'configured': (MODEL_ROOT/'face/model.safetensors').is_file() and (MODEL_ROOT/'detector/yunet.onnx').is_file(), 'loaded': self.face_model is not None},
            'acoustic': {'model': 'superb/wav2vec2-base-superb-er', 'labels': AUDIO_LABELS,
                         'configured': (MODEL_ROOT/'acoustic/pytorch_model.bin').is_file(), 'loaded': self.audio_model is not None}}}

    def face_boxes(self, img):
        """YuNet with confidence filtering and NMS; detection is not emotion inference."""
        import cv2
        scale = min(1.0, 640 / max(img.size))
        size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
        pixels = cv2.cvtColor(np.asarray(img.resize(size)), cv2.COLOR_RGB2BGR)
        if self.detector is None:
            self.detector = cv2.FaceDetectorYN.create(str(MODEL_ROOT/'detector/yunet.onnx'), '', size, .9, .3, 5000)
        self.detector.setInputSize(size)
        _, detected = self.detector.detect(pixels)
        if detected is None:
            return []
        return [tuple(float(v) / scale for v in row[:4]) for row in detected
                if float(row[-1]) >= .9 and min(row[2:4]) / scale >= 32]

    def face(self, frames):
        import cv2
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessorPil
        if not frames: raise ValueError('No camera frames from this recording. Turn on the mirror camera and record again.')
        crops, offsets, skipped, ambiguous = [], [], [], []
        for frame in frames:
            img = image_from_bytes(decode64(frame.image, 512 * 1024))
            faces = self.face_boxes(img)
            if len(faces) > 1:
                ambiguous.append(frame.offset_ms); continue
            if len(faces) != 1:
                skipped.append(frame.offset_ms); continue
            x, y, w, h = map(int, faces[0]); pad = int(max(w, h) * .1)
            crops.append(img.crop((max(0,x-pad), max(0,y-pad), min(img.width,x+w+pad), min(img.height,y+h+pad))))
            offsets.append(frame.offset_ms)
        if not crops:
            if ambiguous:
                raise ValueError('The detector found more than one possible face region in the usable snapshots. This can be a false alarm from a reflection, poster or background. Use a plain background, face the camera and record again. No person count has been confirmed.')
            raise ValueError('No clear frontal face detected. Face the camera in good light and record again.')
        if self.face_model is None:
            self.face_processor = ViTImageProcessorPil.from_pretrained(MODEL_ROOT/'face', local_files_only=True)
            self.face_model = ViTForImageClassification.from_pretrained(MODEL_ROOT/'face', local_files_only=True).eval()
        with torch.inference_mode():
            logits = self.face_model(**self.face_processor(images=crops, return_tensors='pt')).logits
            scores = logits.softmax(-1).mean(0).cpu().numpy()
        result = self.result('face', scores, FACE_LABELS)
        if ambiguous:
            result['warnings'].append(f'{len(ambiguous)} ambiguous snapshot(s) were excluded; only clear single-face snapshots were classified.')
        result.update({'detector': 'OpenCV YuNet 2023mar', 'frames_ambiguous': ambiguous, 'frames_used': len(crops), 'frame_offsets_ms': offsets, 'frames_without_face': skipped,
                       'aggregation': 'Mean of frame softmax scores after YuNet single-face cropping; ambiguous snapshots excluded.'})
        return result

    def acoustic(self, data):
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        from .speech_main import decode_bounded
        if not data: raise ValueError('No audio from this recording. Use Whisper recording and speak again.')
        audio = decode_bounded(decode64(data, 4 * 1024 * 1024))
        duration = len(audio) / 16000
        if duration < 1: raise ValueError('At least one second of speech is required.')
        if duration > 21: raise ValueError('Bonus recordings must be 20 seconds or shorter.')
        features = acoustic_features(audio)
        if features['rms_dbfs'] < -65: raise ValueError('Audio is silent or too quiet. No acoustic prediction was made.')
        if features['voiced_fraction'] < .05: raise ValueError('Insufficient voiced signal. Try a clear sentence in a quiet room.')
        if self.audio_model is None:
            self.audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ROOT/'acoustic', local_files_only=True)
            self.audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ROOT/'acoustic', local_files_only=True).eval()
        # Bound attention memory: contiguous chunks, at most eight seconds each.
        chunks = np.array_split(audio, int(np.ceil(duration / 8)))
        scores = []
        with torch.inference_mode():
            for chunk in chunks:
                inputs = self.audio_processor(chunk, sampling_rate=16000, return_tensors='pt')
                scores.append(self.audio_model(**inputs).logits.softmax(-1)[0].cpu().numpy())
        result = self.result('acoustic', np.average(scores, axis=0, weights=[len(c) for c in chunks]), AUDIO_LABELS)
        result.update({'duration_seconds': round(duration, 2), 'features': features, 'chunks': len(chunks),
                       'aggregation': 'Duration-weighted mean of contiguous chunk softmax scores (up to 8 s each).',
                       'label_names': {'neu': 'neutral', 'hap': 'happy', 'ang': 'angry', 'sad': 'sad'}})
        if features['clipping_fraction'] > .01: result['warnings'].append('Audio clipping may reduce reliability.')
        return result

    def result(self, channel, scores, labels):
        if not np.isfinite(scores).all() or len(scores) != len(labels): raise RuntimeError('Invalid output')
        return {'channel': channel, 'model': self.health()['channels'][channel]['model'],
                'label': labels[int(np.argmax(scores))], 'confidence': float(max(scores)),
                'probabilities': dict(zip(labels, map(float, scores))), 'calibrated': False,
                'warnings': ['Experimental, uncalibrated model scores; not a measure of emotional intensity or inner feelings.']}

    def predict(self, request):
        if not self.lock.acquire(blocking=False): raise HTTPException(429, 'The bonus service is busy. Try again shortly.')
        try:
            results, errors = [], []
            for channel in request.channels:
                start = perf_counter()
                try:
                    if not self.health()['channels'][channel]['configured']: raise ValueError('Model files are missing. Run scripts/prepare_multimodal.py.')
                    result = self.face(request.frames) if channel == 'face' else self.acoustic(request.audio)
                    result['elapsed_ms'] = round((perf_counter()-start)*1000, 1)
                    results.append(result)
                except (ValueError, HTTPException) as error:
                    errors.append({'channel': channel, 'message': str(error.detail) if isinstance(error, HTTPException) else str(error)})
                except Exception:
                    import logging
                    logging.getLogger(__name__).exception('Bonus inference failed for %s', channel)
                    errors.append({'channel': channel, 'message': 'This model is unavailable. Check service dependencies; other results are preserved.'})
            return {'capture_id': request.capture_id, 'results': results, 'errors': errors, 'fusion': None,
                    'note': 'Channels have different label sets and training data. No combined truth score or cross-channel confidence ranking is computed.'}
        finally: self.lock.release()

service = MultimodalService()
