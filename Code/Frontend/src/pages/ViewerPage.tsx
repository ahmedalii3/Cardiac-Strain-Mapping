import Navbar from "../components/Navbar";
import DicomViewer from "../components/Viewer/DicomViewer";
import Chatbot from "../components/Chatbot";

const ViewerPage = () => {
  return (
    <>
      <div className="min-h-screen bg-white">
        <Navbar />
        <DicomViewer />
        <Chatbot />
      </div>
    </>
  );
};

export default ViewerPage;
