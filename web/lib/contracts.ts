export const MODEL_IDS = ['bow_nb', 'tfidf_lr', 'improved_lr', 'bilstm', 'bert', 'llm'] as const;
export type ModelId = typeof MODEL_IDS[number];
export const LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'] as const;
export const NAMES: Record<ModelId, string> = {bow_nb: 'BoW + Naive Bayes', tfidf_lr: 'TF-IDF + LR', improved_lr: 'Improved LR', bilstm: 'BiLSTM', bert: 'Fine-tuned BERT', llm: 'DeepSeek few-shot'};
export const NOTES: Record<ModelId, string> = {bow_nb: 'Word counts', tfidf_lr: 'Sparse baseline', improved_lr: 'Sublinear TF · class weights', bilstm: 'Word order · GloVe', bert: 'Contextual encoder', llm: '18 examples · remote API'};
export type Model = {id: ModelId; available: boolean; enabled: boolean; supports_probabilities: boolean; error?: string | null};
export type Prediction = {model_id: ModelId; label: string; confidence: number | null; probabilities: Record<string, number> | null; calibrated: boolean; temperature: number | null; elapsed_ms: number; warnings: string[]; truncated: boolean; provider: string; served_model?: string; prompt_id?: string; n_shots?: number; usage?: {prompt_tokens: number; completion_tokens: number; n_requests: number}};
export type Comparison = {text: string; results: Prediction[]; errors: {model_id: ModelId; message: string; code: string}[]; all_agree: boolean | null};
export function validateInput(text: string, models: string[]): string | null {
  if (!text.trim()) return 'Say something first, or type a sentence below.';
  if (text.length > 2000) return 'Please shorten your transcript to 2,000 characters.';
  if (!/[A-Za-z]/.test(text)) return 'Use an English sentence. These models were evaluated on English text.';
  if (/[\x00-\x08\x0b\x0c\x0e-\x1f]/.test(text)) return 'Please remove unsupported control characters.';
  if (!models.length) return 'Choose at least one model.';
  if (models.length > 6 || new Set(models).size !== models.length || models.some(m => !MODEL_IDS.includes(m as ModelId))) return 'Choose supported models, once each.';
  return null;
}
export function isComparison(value: unknown): value is Comparison {
  if (!value || typeof value !== 'object') return false;
  const v = value as Comparison;
  return typeof v.text === 'string' && Array.isArray(v.results) && Array.isArray(v.errors) && (v.all_agree === null || typeof v.all_agree === 'boolean') &&
    v.results.every(r => MODEL_IDS.includes(r.model_id) && LABELS.includes(r.label as typeof LABELS[number]) && Array.isArray(r.warnings) &&
      (r.model_id === 'llm' ? r.probabilities === null && r.confidence === null :
        typeof r.confidence === 'number' && r.confidence >= 0 && r.confidence <= 1 && r.probabilities !== null && LABELS.every(l => Number.isFinite(r.probabilities?.[l]) && r.probabilities![l] >= 0 && r.probabilities![l] <= 1))) &&
    v.errors.every(e => MODEL_IDS.includes(e.model_id) && typeof e.message === 'string');
}
