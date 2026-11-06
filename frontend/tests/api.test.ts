import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { validateFile, MAX_FILE_SIZE } from '../src/lib/upload';

// Synthetic fixtures exercise the contract only; they are never imported by the app.
const result = { predicted_class: 'book', confidence: 0.7, probabilities: { book: 0.7, bottle: 0.1, clock: 0.1, keyboard: 0.1 }, latency_ms: 12 };
const fixture = () => ({ results: { dinov2: structuredClone(result) }, nearest_neighbors: [{ image_url: '/api/images/book.jpg', class_name: 'book', similarity: 0.8 }], device: 'cpu' });
beforeEach(() => { vi.resetModules(); vi.stubEnv('VITE_API_BASE_URL', 'http://127.0.0.1:8000'); vi.stubEnv('VITE_API_TIMEOUT_MS', '1000'); });
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); vi.useRealTimers(); });

describe('response validation', () => {
  it('accepts real-shaped responses and resolves images against the backend', async () => {
    const { validateResponse, resolveImageUrl } = await import('../src/lib/api');
    expect(validateResponse(fixture(), 'dinov2').device).toBe('cpu');
    expect(resolveImageUrl('/api/images/book.jpg')).toBe('http://127.0.0.1:8000/api/images/book.jpg');
  });
  it('requires both model results for a comparison', async () => {
    const { validateResponse } = await import('../src/lib/api');
    expect(() => validateResponse(fixture(), 'both')).toThrow('invalid prediction');
    const data = { ...fixture(), results: { dinov2: result, resnet18: result } };
    expect(validateResponse(data, 'both').results.resnet18).toBeDefined();
  });
  it.each([
    ['out-of-range confidence', (data: ReturnType<typeof fixture>) => { data.results.dinov2.confidence = 98; }],
    ['invalid probability sum', (data: ReturnType<typeof fixture>) => { data.results.dinov2.probabilities.clock = 0.9; }],
    ['mismatched confidence', (data: ReturnType<typeof fixture>) => { data.results.dinov2.confidence = 0.2; }],
    ['negative latency', (data: ReturnType<typeof fixture>) => { data.results.dinov2.latency_ms = -1; }],
    ['unknown category', (data: ReturnType<typeof fixture>) => { data.results.dinov2.predicted_class = 'chair'; }],
    ['nonfinite similarity', (data: ReturnType<typeof fixture>) => { data.nearest_neighbors[0].similarity = NaN; }],
    ['empty device', (data: ReturnType<typeof fixture>) => { data.device = ''; }],
  ])('rejects %s', async (_, mutate) => {
    const { validateResponse } = await import('../src/lib/api'); const data = fixture(); mutate(data);
    expect(() => validateResponse(data, 'dinov2')).toThrow('invalid prediction');
  });
  it('rejects executable image URLs', async () => {
    const { validateResponse } = await import('../src/lib/api'); const data = fixture(); data.nearest_neighbors[0].image_url = 'javascript:alert(1)';
    expect(() => validateResponse(data, 'dinov2')).toThrow('unsupported image URL');
  });
});

describe('prediction client', () => {
  const file = new File(['test'], 'book.png', { type: 'image/png' });
  it('sends multipart fields and validates the response', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(fixture()), { status: 200 })); vi.stubGlobal('fetch', fetch);
    const { predict } = await import('../src/lib/api'); await predict(file, 'dinov2');
    const [url, options] = fetch.mock.calls[0];
    expect(url).toBe('http://127.0.0.1:8000/api/predict'); expect(options.body.get('model')).toBe('dinov2'); expect(options.body.get('image').name).toBe('book.png'); expect(options.headers).toBeUndefined();
  });
  it('does not make a request when no backend is configured', async () => {
    vi.stubEnv('VITE_API_BASE_URL', ''); const fetch = vi.fn(); vi.stubGlobal('fetch', fetch); const { predict } = await import('../src/lib/api');
    await expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'disconnected' }); expect(fetch).not.toHaveBeenCalled();
  });
  it('reports network failures without creating fallback predictions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch'))); const { predict } = await import('../src/lib/api');
    await expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'disconnected' });
  });
  it.each([413, 422, 500])('reports server error %s', async status => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('error', { status }))); const { predict } = await import('../src/lib/api');
    await expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'server' });
  });
  it('rejects non-JSON success responses', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('<html>Wrong endpoint</html>'))); const { predict } = await import('../src/lib/api');
    await expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'invalid' });
  });
  it('shows the actual backend validation error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'Image exceeds the 25 megapixel limit.' }), { status: 413 })));
    const { predict } = await import('../src/lib/api');
    await expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'server', message: 'Image exceeds the 25 megapixel limit.' });
  });
  it('checks readiness of all backend components', async () => {
    const health = { status: 'ok', device: 'cpu', resnet18_loaded: true, dinov2_loaded: true, classifier_loaded: true, neighbors_ready: true, training_images: 280, error: null };
    const fetch = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify(health))).mockResolvedValueOnce(new Response(JSON.stringify({ ...health, neighbors_ready: false })));
    vi.stubGlobal('fetch', fetch);
    const { checkHealth } = await import('../src/lib/api');
    const signal = new AbortController().signal;
    expect((await checkHealth(signal)).training_images).toBe(280);
    await expect(checkHealth(signal)).rejects.toMatchObject({ kind: 'disconnected' });
    expect(fetch.mock.calls[0][0]).toBe('http://127.0.0.1:8000/api/health');
  });
  it('aborts a stalled request after the configured timeout', async () => {
    vi.useFakeTimers(); vi.stubGlobal('fetch', vi.fn((_url, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))))));
    const { predict } = await import('../src/lib/api'); const pending = expect(predict(file, 'dinov2')).rejects.toMatchObject({ kind: 'timeout' });
    await vi.advanceTimersByTimeAsync(1001); await pending;
  });
  it('preserves caller cancellation without turning it into a network failure', async () => {
    vi.stubGlobal('fetch', vi.fn((_url, options) => new Promise((_resolve, reject) => options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError'))))));
    const { predict } = await import('../src/lib/api'); const controller = new AbortController();
    const pending = expect(predict(file, 'dinov2', controller.signal)).rejects.toMatchObject({ name: 'AbortError' });
    controller.abort(); await pending;
  });
});

describe('upload validation', () => {
  it('accepts supported formats and rejects empty, mismatched, or oversized files', () => {
    expect(validateFile(new File(['image'], 'test.jpeg', { type: 'image/jpeg' }))).toBeNull();
    expect(validateFile(new File([], 'test.png', { type: 'image/png' }))).toContain('empty');
    expect(validateFile(new File(['image'], 'test.svg', { type: 'image/svg+xml' }))).toContain('Choose a JPG');
    expect(validateFile(new File(['image'], 'test.gif', { type: 'image/png' }))).toContain('Choose a JPG');
    expect(validateFile(new File([new Uint8Array(MAX_FILE_SIZE + 1)], 'test.webp', { type: 'image/webp' }))).toContain('too large');
  });
});
