import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();
  const isDashboard = location.pathname === "/";

  return (
    <nav className="navbar">
      <Link to="/" className="navbar-brand" id="nav-brand">
        <div className="navbar-logo-icon">⚗️</div>
        <div className="navbar-brand-name">Sci<span>Mine</span></div>
      </Link>

      <div className="navbar-links">
        <Link to="/" className={`navbar-link ${isDashboard ? "navbar-link-active" : ""}`} id="nav-home">
          Dashboard
        </Link>
        <span className="navbar-badge">NLP · T5 · TF-IDF</span>
      </div>
    </nav>
  );
}