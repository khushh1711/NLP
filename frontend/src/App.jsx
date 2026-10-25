import { BrowserRouter, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Dashboard      from "./pages/Dashboard";
import WorkspaceDetail from "./pages/WorkspaceDetail";
import PaperDetail    from "./pages/PaperDetail";
import ComparisonView from "./pages/ComparisonView";
import InsightsView   from "./pages/InsightsView";

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"                                     element={<Dashboard />} />
        <Route path="/workspace/:workspaceId"               element={<WorkspaceDetail />} />
        <Route path="/workspace/:workspaceId/comparison"    element={<ComparisonView />} />
        <Route path="/workspace/:workspaceId/insights"      element={<InsightsView />} />
        <Route path="/paper/:paperId"                       element={<PaperDetail />} />
      </Routes>
    </BrowserRouter>
  );
}