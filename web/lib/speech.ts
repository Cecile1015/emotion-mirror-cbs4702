// The browser owns transcription. This module never calls the emotion API.
export type RecognitionResult = {isFinal: boolean; 0: {transcript: string}};
export type RecognitionEvent = {results: ArrayLike<RecognitionResult>};
export type Recognition = {lang: string; continuous: boolean; interimResults: boolean; maxAlternatives: number; onstart: (() => void) | null; onend: (() => void) | null; onresult: ((e: RecognitionEvent) => void) | null; onerror: ((e: {error: string}) => void) | null; start(): void; stop(): void; abort(): void};
export type SpeechConstructor = new () => Recognition;
export function joinTranscript(base: string, results: ArrayLike<RecognitionResult>) {
  const final: string[] = [], interim: string[] = [];
  for (let i = 0; i < results.length; i++) (results[i].isFinal ? final : interim).push(results[i][0].transcript.trim());
  return {final: [base.trim(), ...final].filter(Boolean).join(' '), interim: interim.filter(Boolean).join(' ')};
}
export const speechError = (code: string) => ({
  'not-allowed': 'Microphone access was denied. Allow it in browser settings, then try again. You can also type below.',
  'service-not-allowed': 'Speech recognition is blocked by this browser or its settings. Try Chrome or Edge, or type below.',
  'audio-capture': 'No working microphone was found. Check your input device, or type below.',
  'no-speech': 'No speech was detected. Move closer to the microphone and try again, or type below.',
  'network': 'The browser’s speech service could not connect. Check your connection, try again, or type below.',
  'language-not-supported': 'English speech recognition is unavailable in this browser. You can type below.',
}[code] || 'Speech recognition stopped. Your transcript is preserved; try again or continue by typing.');
