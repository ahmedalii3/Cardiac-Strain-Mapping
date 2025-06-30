import Chatbot from "../components/Chatbot";
import Features from "../components/Features";
import Footer from "../components/Footer";
import Header from "../components/Header";
import Importance from "../components/Importance";
import Navbar from "../components/Navbar";

const LangingPage = () => {
  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <Header />
      <Features />
      <Importance />
      <Chatbot />
      <Footer />
    </div>
  );
};

export default LangingPage;
