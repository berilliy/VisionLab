import { NavLink, Outlet } from 'react-router-dom';
import { useInference } from '../context/InferenceContext';

export function Header() {
  const { connection } = useInference();
  return (
    <header className="header">
      <NavLink className="brand" to="/">VisionLab</NavLink>
      <nav aria-label="Main navigation">
        <NavLink to="/" end>Classification</NavLink>
        <NavLink to="/comparison">Comparison</NavLink>
      </nav>
      <span className={`connection ${connection}`}>
        <span className="dot" />
        {connection === 'connected' ? 'Backend ready' : connection === 'unverified' ? 'Checking backend' : 'Backend not connected'}
      </span>
    </header>
  );
}

export default function Layout() {
  return (
    <>
      <a className="skip-link" href="#main">Skip to content</a>
      <Header />
      <main id="main" tabIndex={-1}><Outlet /></main>
    </>
  );
}
