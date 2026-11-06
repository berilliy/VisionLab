import { CLASSES, type HealthResponse, type ModelSelection, type PredictionResponse } from './types';

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? '').trim().replace(/\/+$/, '');
const configuredTimeout = Number(import.meta.env.VITE_API_TIMEOUT_MS);
const TIMEOUT_MS = Number.isFinite(configuredTimeout) && configuredTimeout > 0 ? configuredTimeout : 60000;
export class ApiError extends Error {
  constructor(message: string, public kind: 'disconnected' | 'timeout' | 'server' | 'invalid') { super(message); }
}

export async function checkHealth(signal: AbortSignal): Promise<HealthResponse> {
  if (!API_BASE_URL) throw new ApiError('Backend not connected.', 'disconnected');
  const response = await fetch(`${API_BASE_URL}/api/health`, { signal });
  const data: unknown = await response.json();
  if (!response.ok || !record(data) || data.status !== 'ok' ||
      !data.resnet18_loaded || !data.dinov2_loaded || !data.classifier_loaded || !data.neighbors_ready) {
    throw new ApiError(record(data) && typeof data.error === 'string' ? data.error : 'Backend is not ready.', 'disconnected');
  }
  return data as unknown as HealthResponse;
}
const record = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const probability = (value: unknown): value is number => finite(value) && value >= 0 && value <= 1;
const validClass = (value: unknown) => CLASSES.includes(value as (typeof CLASSES)[number]);

export function resolveImageUrl(path: string, base = API_BASE_URL): string {
  const url = new URL(path, base ? `${base}/` : window.location.origin);
  if (!['http:', 'https:'].includes(url.protocol)) throw new ApiError('The backend returned an unsupported image URL.', 'invalid');
  return url.href;
}

export function validateResponse(data: unknown, model: ModelSelection): PredictionResponse {
  const invalid = () => { throw new ApiError('The backend returned an invalid prediction response. Check the API contract.', 'invalid'); };
  if (!record(data) || !record(data.results) || typeof data.device !== 'string' || !data.device.trim() || !Array.isArray(data.nearest_neighbors)) return invalid();
  const expected = model === 'both' ? ['dinov2', 'resnet18'] : [model];
  if (expected.some(key => !data.results || !record(data.results) || !data.results[key])) return invalid();
  for (const [key, result] of Object.entries(data.results)) {
    if (!['dinov2', 'resnet18'].includes(key) || !record(result) || !validClass(result.predicted_class) || !probability(result.confidence) || !finite(result.latency_ms) || result.latency_ms < 0 || !record(result.probabilities)) return invalid();
    const probabilities = result.probabilities;
    if (CLASSES.some(cls => !probability(probabilities[cls]))) return invalid();
    const sum = CLASSES.reduce((total, cls) => total + Number(probabilities[cls]), 0);
    if (Math.abs(sum - 1) > 0.02 || Math.abs(Number(probabilities[String(result.predicted_class)]) - result.confidence) > 0.005) return invalid();
  }
  if (data.runtime !== undefined && typeof data.runtime !== 'string') return invalid();
  for (const neighbor of data.nearest_neighbors) {
    if (!record(neighbor) || typeof neighbor.image_url !== 'string' || !neighbor.image_url.trim() || !validClass(neighbor.class_name) || !finite(neighbor.similarity) || neighbor.similarity < -1 || neighbor.similarity > 1) return invalid();
    resolveImageUrl(neighbor.image_url);
  }
  return data as unknown as PredictionResponse;
}

export async function predict(file: File, model: ModelSelection, signal?: AbortSignal): Promise<PredictionResponse> {
  if (!API_BASE_URL) throw new ApiError('Backend not connected. Set VITE_API_BASE_URL to your Python API address and restart the frontend.', 'disconnected');
  try { const url = new URL(API_BASE_URL); if (!['http:', 'https:'].includes(url.protocol)) throw new Error(); }
  catch { throw new ApiError('Backend not connected. VITE_API_BASE_URL must be an HTTP or HTTPS address.', 'disconnected'); }
  const controller = new AbortController();
  let timedOut = false;
  const cancel = () => controller.abort();
  signal?.addEventListener('abort', cancel, { once: true });
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(() => { timedOut = true; controller.abort(); }, TIMEOUT_MS);
  const body = new FormData();
  body.append('image', file);
  body.append('model', model);
  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, { method: 'POST', body, signal: controller.signal });
    if (!response.ok) {
      let detail = '';
      try {
        const body: unknown = await response.json();
        if (record(body) && typeof body.detail === 'string') detail = body.detail.slice(0, 500);
      } catch { /* Use status-based guidance if the response is not JSON. */ }
      if ([404, 502, 503, 504].includes(response.status)) throw new ApiError(`Backend not connected or unavailable (HTTP ${response.status}). ${detail || 'Check that the Python server is running.'}`, 'disconnected');
      throw new ApiError(detail || `The server could not analyze this image (HTTP ${response.status}). ${response.status === 413 ? 'Try a smaller file.' : response.status === 429 ? 'Another image is being processed; retry shortly.' : 'Please try again.'}`, 'server');
    }
    let data: unknown;
    try { data = await response.json(); } catch (error) { if (controller.signal.aborted) throw error; throw new ApiError('The API returned unreadable data. Check that /api/predict returns JSON.', 'invalid'); }
    return validateResponse(data, model);
  } catch (error) {
    if (timedOut) throw new ApiError(`The request timed out after ${Math.round(TIMEOUT_MS / 1000)} seconds. The model may still be starting; please retry.`, 'timeout');
    if (signal?.aborted) throw error;
    if (error instanceof ApiError) throw error;
    throw new ApiError('Backend not connected. Check the Python server, network connection, and backend CORS settings, then retry.', 'disconnected');
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', cancel);
  }
}
