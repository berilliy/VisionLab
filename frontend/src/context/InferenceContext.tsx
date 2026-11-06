import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from 'react';
import { API_BASE_URL, ApiError, checkHealth, predict } from '../lib/api';
import { validateFile } from '../lib/upload';
import type { ModelSelection, PredictionResponse } from '../lib/types';

function useInferenceState() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<ModelSelection>('dinov2');
  const [response, setResponse] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [connection, setConnection] = useState<'unverified' | 'connected' | 'disconnected'>(API_BASE_URL ? 'unverified' : 'disconnected');
  const request = useRef<AbortController | null>(null);
  const version = useRef(0);
  const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; request.current?.abort(); }; }, []);
  useEffect(() => {
    if (!API_BASE_URL) return;
    const controller = new AbortController();
    let active = true;
    const token = version.current;
    const timeout = setTimeout(() => controller.abort(), 5000);
    checkHealth(controller.signal)
      .then(() => { if (active && mounted.current && token === version.current) setConnection('connected'); })
      .catch(() => { if (active && mounted.current && token === version.current) setConnection('disconnected'); })
      .finally(() => clearTimeout(timeout));
    return () => { active = false; clearTimeout(timeout); controller.abort(); };
  }, []);
  useEffect(() => {
    if (!file) { setPreview(null); return; }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  function cancel() { version.current++; request.current?.abort(); setLoading(false); }
  function changeModel(next: ModelSelection) { cancel(); setModel(next); setResponse(null); setError(null); }
  async function selectFile(next: File | null) {
    cancel();
    setResponse(null); setError(null); setFile(null);
    if (!next) return;
    const problem = validateFile(next);
    if (problem) { setError(problem); return; }
    const token = version.current;
    const url = URL.createObjectURL(next);
    try {
      await new Promise<void>((resolve, reject) => { const img = new Image(); img.onload = () => resolve(); img.onerror = () => reject(new Error('This file could not be decoded as an image. Please choose another file.')); img.src = url; });
      if (mounted.current && token === version.current) setFile(next);
    } catch (err) { if (mounted.current && token === version.current) setError((err as Error).message); }
    finally { URL.revokeObjectURL(url); }
  }
  async function analyze() {
    if (!file || loading) return;
    cancel();
    const token = version.current;
    const controller = new AbortController(); request.current = controller;
    setLoading(true); setError(null); setResponse(null);
    try {
      const result = await predict(file, model, controller.signal);
      if (token === version.current && mounted.current) { setResponse(result); setConnection('connected'); }
    } catch (err) {
      if (token === version.current && mounted.current) {
        setError(err instanceof Error ? err.message : 'Analysis failed. Please retry.');
        setConnection(err instanceof ApiError && err.kind === 'disconnected' ? 'disconnected' : 'unverified');
      }
    } finally { if (token === version.current && mounted.current) setLoading(false); }
  }
  return { file, preview, model, response, loading, error, connection, selectFile, changeModel, analyze, cancel };
}
const Context = createContext<ReturnType<typeof useInferenceState> | null>(null);
export function InferenceProvider({ children }: { children: ReactNode }) { const value = useInferenceState(); return <Context.Provider value={value}>{children}</Context.Provider>; }
export function useInference() { const context = useContext(Context); if (!context) throw new Error('InferenceProvider is required'); return context; }
