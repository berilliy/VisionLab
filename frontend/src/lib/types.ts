export const CLASSES = ['book', 'bottle', 'clock', 'keyboard'] as const;
export type ObjectClass = (typeof CLASSES)[number];
export type ModelId = 'dinov2' | 'resnet18';
export type ModelSelection = ModelId | 'both';
export interface PredictionResult {
  predicted_class: ObjectClass;
  confidence: number;
  probabilities: Record<ObjectClass, number>;
  latency_ms: number;
}
export interface NearestNeighbor {
  image_url: string;
  class_name: ObjectClass;
  similarity: number;
}
export interface PredictionResponse {
  results: Partial<Record<ModelId, PredictionResult>>;
  nearest_neighbors: NearestNeighbor[];
  device: string;
  runtime?: string;
}
export interface HealthResponse {
  status: 'ok' | 'error' | 'loading';
  device: 'cpu';
  resnet18_loaded: boolean;
  dinov2_loaded: boolean;
  classifier_loaded: boolean;
  neighbors_ready: boolean;
  training_images: number;
  error: string | null;
}
export const MODEL_NAMES: Record<ModelId, string> = { dinov2: 'DINOv2', resnet18: 'ResNet18' };
