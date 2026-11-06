import { useInference } from '../context/InferenceContext';
import { UploadZone, ModelSelector } from '../components/UploadZone';
import { PredictionCard } from '../components/PredictionCard';
import { NearestNeighborsGallery } from '../components/NearestNeighborsGallery';
import { EmptyState, ErrorState, LoadingState, SectionHeading } from '../components/ui';
import type { ModelId } from '../lib/types';

export default function Classification() {
  const state = useInference();
  return (
    <div className="page-stack">
      <div className="page-title">
        <h1>Object classification</h1>
        <p>Book, bottle, clock, keyboard</p>
      </div>
      <div className="classification-grid">
        <section className="panel upload-panel">
          <SectionHeading title="Image" />
          <UploadZone file={state.file} preview={state.preview} onSelect={state.selectFile} />
          <ModelSelector value={state.model} onChange={state.changeModel} />
          <button className="button primary analyze-button" disabled={!state.file || state.loading} onClick={state.analyze}>
            {state.loading ? 'Analyzing...' : 'Analyze image'}
          </button>
          <p className="micro-copy mt-3">Photos are processed in memory and are not stored.</p>
        </section>
        <section className="panel results-panel" aria-busy={state.loading} aria-live="polite">
          <SectionHeading title="Result" />
          {state.loading ? <LoadingState onCancel={state.cancel} /> : state.error ? (
            <ErrorState message={state.error} retry={state.file ? state.analyze : undefined} />
          ) : state.response ? (
            <div className="prediction-stack">
              {(['dinov2', 'resnet18'] as ModelId[]).filter(model => state.response?.results[model]).map(model => (
                <PredictionCard key={model} model={model} result={state.response!.results[model]!} device={state.response!.device} runtime={state.response!.runtime} />
              ))}
            </div>
          ) : (
            <EmptyState title="No prediction yet" description={state.file ? 'Press Analyze image to get a result.' : 'Upload an image to start.'} />
          )}
        </section>
      </div>
      {!!state.response?.nearest_neighbors.length && (
        <details className="panel neighbors-details">
          <summary>Similar training images</summary>
          <NearestNeighborsGallery input={state.preview} neighbors={state.response.nearest_neighbors} />
        </details>
      )}
    </div>
  );
}
