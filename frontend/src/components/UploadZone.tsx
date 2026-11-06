import { useRef, useState } from 'react';
import { ImagePlus, Upload, X, RefreshCw, FileImage } from 'lucide-react';
import type { ModelSelection } from '../lib/types';

export function ImagePreview({ src, name }: { src: string; name: string }) { return <img className="image-preview" src={src} alt={`Uploaded input: ${name}`} />; }
export function UploadZone({ file, preview, onSelect }: { file: File | null; preview: string | null; onSelect: (file: File | null) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const dragDepth = useRef(0);
  return <div><input ref={input} type="file" accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp" className="sr-only" aria-label="Upload household object image" onChange={event => { const selected = event.target.files?.[0]; if (selected) onSelect(selected); event.target.value = ''; }} />
    <div className={`upload-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-image' : ''}`} onDragEnter={event => { event.preventDefault(); dragDepth.current++; setDragging(true); }} onDragLeave={event => { event.preventDefault(); dragDepth.current--; if (dragDepth.current === 0) setDragging(false); }} onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); dragDepth.current = 0; setDragging(false); const selected = event.dataTransfer.files[0]; if (selected) onSelect(selected); }}>
      {preview && file ? <><ImagePreview src={preview} name={file.name} /><div className="preview-actions"><button className="icon-button" aria-label="Replace image" onClick={() => input.current?.click()}><RefreshCw size={17} /></button><button className="icon-button" aria-label="Remove image" onClick={() => onSelect(null)}><X size={18} /></button></div></> : <><div className="upload-symbol"><ImagePlus size={34} strokeWidth={1.4} /></div><h3>Drop an image here</h3><button className="button secondary" onClick={() => input.current?.click()}><Upload size={16} />Browse files</button><small>JPG, PNG or WEBP · Up to 10 MB</small></>}
    </div>{file && <div className="file-info"><FileImage size={16} /><span title={file.name}>{file.name}</span><small>{(file.size / 1024 / 1024).toFixed(2)} MB</small></div>}
  </div>;
}
export function ModelSelector({ value, onChange }: { value: ModelSelection; onChange: (value: ModelSelection) => void }) {
  return <fieldset className="model-selector"><legend>Model</legend><div className="model-options">{([{ id: 'dinov2', title: 'DINOv2', detail: 'Vision transformer' }, { id: 'resnet18', title: 'ResNet18', detail: 'Convolutional network' }, { id: 'both', title: 'Compare both', detail: 'Side-by-side inference' }] as const).map(item => <label key={item.id} className={value === item.id ? 'selected' : ''}><input type="radio" name="model" value={item.id} checked={value === item.id} onChange={() => onChange(item.id)} /><span><strong>{item.title}</strong></span></label>)}</div></fieldset>;
}
