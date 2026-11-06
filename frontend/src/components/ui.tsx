import { AlertCircle, ScanLine, LoaderCircle, type LucideIcon } from 'lucide-react';
import type { ReactNode } from 'react';

export function SectionHeading({ title, detail, children }: { title: string; detail?: string; children?: ReactNode }) {
  return <div className="section-heading"><div><h2>{title}</h2>{detail && <p>{detail}</p>}</div>{children}</div>;
}
export function EmptyState({ title = 'No result yet', description = 'Upload an image to start.', icon: Icon = ScanLine, children }: { title?: string; description?: string; icon?: LucideIcon; children?: ReactNode }) {
  return <div className="empty-state"><Icon size={28} strokeWidth={1.5} /><h3>{title}</h3><p>{description}</p>{children}</div>;
}
export function LoadingState({ onCancel }: { onCancel: () => void }) {
  return <div className="empty-state" role="status"><LoaderCircle className="spin" size={28} /><h3>Analyzing image...</h3><button className="button secondary" onClick={onCancel}>Cancel</button></div>;
}
export function ErrorState({ message, retry }: { message: string; retry?: () => void }) {
  return <div className="error-state" role="alert"><AlertCircle size={20} /><div><strong>Analysis failed</strong><p>{message}</p>{retry && <button className="text-button" onClick={retry}>Retry</button>}</div></div>;
}
