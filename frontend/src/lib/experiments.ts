// Historical measurements supplied with the assignment. Never used for live inference.
export const experiments = [
  { name: 'HOG + SVM', accuracy: 68.33, precision: 69.24, recall: 68.33, f1: 67.16, color: '#64748B' },
  { name: 'ResNet18', accuracy: 100, precision: 100, recall: 100, f1: 100, color: '#6366F1' },
  { name: 'DINOv2', accuracy: 100, precision: 100, recall: 100, f1: 100, color: '#22D3EE' },
];
export const latency = [{ name: 'ResNet18', value: 45.95, color: '#6366F1' }, { name: 'DINOv2', value: 164.2, color: '#22D3EE' }];
export const unseenBooks = [{ name: 'ResNet18', value: 70, color: '#6366F1' }, { name: 'DINOv2', value: 100, color: '#22D3EE' }];
