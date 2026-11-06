import { useState } from 'react';
import { GeneralizationChart, LatencyChart, ModelComparisonChart } from '../components/ModelComparisonChart';
import { SectionHeading } from '../components/ui';
import { experiments } from '../lib/experiments';

export default function Comparison() {
  const [metric, setMetric] = useState<'accuracy' | 'f1'>('accuracy');
  return (
    <div className="page-stack">
      <div className="page-title">
        <h1>Model comparison</h1>
        <p>Existing experiment results. Test set: 60 images across four classes.</p>
      </div>
      <section className="panel table-panel">
        <div className="table-scroll">
          <table>
            <caption className="sr-only">Original test set results</caption>
            <thead><tr><th scope="col">Model</th><th scope="col">Accuracy</th><th scope="col">Precision</th><th scope="col">Recall</th><th scope="col">F1</th></tr></thead>
            <tbody>{experiments.map(item => (
              <tr key={item.name}>
                <th scope="row"><span className="model-dot" style={{ background: item.color }} />{item.name}</th>
                <td>{item.accuracy.toFixed(2)}%</td><td>{item.precision.toFixed(2)}%</td><td>{item.recall.toFixed(2)}%</td><td>{item.f1.toFixed(2)}%</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
        <p className="micro-copy">The test set is small; accuracy on new images may be lower.</p>
      </section>
      <div className="two-column">
        <section className="panel chart-panel">
          <SectionHeading title="Test performance" />
          <div className="segmented-control" aria-label="Evaluation metric">
            {(['accuracy', 'f1'] as const).map(value => (
              <button key={value} aria-pressed={metric === value} className={metric === value ? 'selected' : ''} onClick={() => setMetric(value)}>
                {value === 'accuracy' ? 'Accuracy' : 'F1 score'}
              </button>
            ))}
          </div>
          <ModelComparisonChart metric={metric} />
        </section>
        <section className="panel chart-panel">
          <SectionHeading title="CPU latency" detail="Preliminary average of 10 runs, including preprocessing." />
          <LatencyChart />
        </section>
      </div>
      <section className="panel chart-panel book-evaluation">
        <SectionHeading title="Additional book test" detail="10 unseen book images, separate from the original test set." />
        <GeneralizationChart />
        <p className="micro-copy">DINOv2: 10/10 correct; ResNet18: 7/10. This evaluates books only, not all four classes.</p>
      </section>
      <p className="micro-copy">ResNet18 was faster on CPU. DINOv2 scored higher on the small additional book test.</p>
    </div>
  );
}
