import { useEffect } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Layout from './components/Layout';
import { InferenceProvider } from './context/InferenceContext';
import Classification from './pages/Classification';
import Comparison from './pages/Comparison';

function NavigationEffects() {
  const { pathname } = useLocation();
  useEffect(() => {
    document.title = `${pathname === '/comparison' ? 'Model comparison' : 'Classification'} | VisionLab`;
    window.scrollTo({ top: 0, behavior: 'instant' });
    document.getElementById('main')?.focus({ preventScroll: true });
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <BrowserRouter>
      <InferenceProvider>
        <NavigationEffects />
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<Classification />} />
            <Route path="classify" element={<Navigate to="/" replace />} />
            <Route path="comparison" element={<Comparison />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </InferenceProvider>
    </BrowserRouter>
  );
}
