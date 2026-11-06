import { useState } from 'react';
import { ImageOff } from 'lucide-react';
import type { NearestNeighbor } from '../lib/types';
import { resolveImageUrl } from '../lib/api';

function GalleryImage({ src, alt }: { src: string; alt: string }) {
  const [failed, setFailed] = useState(false);
  return failed ? <div className="missing-image"><ImageOff size={24} /><span>Image unavailable</span></div> : <img src={src} alt={alt} onError={() => setFailed(true)} loading="lazy" />;
}
export function NearestNeighborsGallery({ input, neighbors }: { input: string | null; neighbors: NearestNeighbor[] }) {
  return (
    <div className="gallery-panel">
      <div className="neighbor-grid">
        {input && <figure className="neighbor input-neighbor"><GalleryImage key={input} src={input} alt="Uploaded image" /><figcaption><strong>Input</strong></figcaption></figure>}
        {neighbors.slice(0, 3).map((neighbor, i) => (
          <figure className="neighbor" key={`${neighbor.image_url}-${i}`}>
            <GalleryImage src={resolveImageUrl(neighbor.image_url)} alt={`Training example: ${neighbor.class_name}`} />
            <figcaption><strong className="capitalize">{neighbor.class_name}</strong><span>Cosine similarity <b>{neighbor.similarity.toFixed(3)}</b></span></figcaption>
          </figure>
        ))}
      </div>
      <p className="micro-copy">DINOv2 feature similarity, not classification confidence or a causal explanation.</p>
    </div>
  );
}
