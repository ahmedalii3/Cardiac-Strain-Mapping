import Chatbot from "../components/Chatbot";
import Navbar from "../components/Navbar";
import ImageViewer from "../components/Results/ImageViewer";

const ResultsPage = () => {
  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar />
      <ImageViewer />
      <Chatbot />
    </div>
  );
};

export default ResultsPage;
