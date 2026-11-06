import { Cpu, Timer } from 'lucide-react';
import { CLASSES, MODEL_NAMES, type ModelId, type PredictionResult } from '../lib/types';

export function ConfidenceChart({ probabilities }: { probabilities: PredictionResult['probabilities'] }) {
  return (
    <div className="confidence-chart">
      {CLASSES.map(cls => (
        <div key={cls} className="probability-row">
          <div><span className="capitalize">{cls}</span><span>{(probabilities[cls] * 100).toFixed(1)}%</span></div>
          <div className="probability-track" role="meter" aria-label={`${cls} model probability`} aria-valuenow={probabilities[cls] * 100} aria-valuemin={0} aria-valuemax={100}>
            <span style={{ width: `${probabilities[cls] * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
export function PredictionCard({ model, result, device, runtime }: { model: ModelId; result: PredictionResult; device: string; runtime?: string }) {
  return (
    <article className="prediction-card">
      <div className="model-name">{MODEL_NAMES[model]}</div>
      <div className="prediction-head">
        <div><div className="tiny-label">Prediction</div><h3 className="capitalize">{result.predicted_class}</h3></div>
        <div className="confidence-value"><strong>{(result.confidence * 100).toFixed(1)}<span>%</span></strong><small>Confidence</small></div>
      </div>
      <ConfidenceChart probabilities={result.probabilities} />
      <div className="inference-meta">
        <span><Timer size={15} />{result.latency_ms.toFixed(2)} ms</span>
        <span><Cpu size={15} />{device}{runtime ? ` · ${runtime}` : ''}</span>
      </div>
      <p className="micro-copy">Model confidence is not test accuracy.</p>
    </article>
  );
}
