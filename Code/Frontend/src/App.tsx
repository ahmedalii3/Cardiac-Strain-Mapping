import "./App.css";
import LangingPage from "./pages/LandingPage";
import Results from "./pages/ResultsPage";
import ViewerPage from "./pages/ViewerPage";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

function App() {
  return (
    <>
      <div className="bg-gray-900 text-white min-h-screen flex-col flex-1">
        <Router>
          <Routes>
            <Route path="/" element={<LangingPage />} />
            <Route path="/viewer" element={<ViewerPage />} />
            <Route path="*" element={<LangingPage />} />
            <Route path="/results" element={<Results />} />
          </Routes>
        </Router>
      </div>
    </>
  );
}

export default App;
