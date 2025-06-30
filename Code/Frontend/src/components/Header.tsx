import { useNavigate } from "react-router-dom";

const Header = () => {
  const navigate = useNavigate();
  const handleFeaturesClick = () => {
    const featuresSection = document.getElementById("features");
    if (featuresSection) {
      featuresSection.scrollIntoView({ behavior: "smooth" });
    }
  };
  return (
    <>
      <header className="relative pt-16 overflow-hidden">
        <div className="absolute inset-0">
          <img
            src="src/assets/Header-Img.png"
            alt="Cardiac MRI Background"
            className="w-full h-full object-cover filter brightness-100"
          />
          <div className="absolute inset-0 bg-gradient-to-r from-blue-900/90 to-gray-900/90" />
        </div>

        <div className="relative max-w-7xl mx-auto py-24 px-4 sm:px-6 lg:px-8">
          <div className="text-center">
            <h1 className="text-4xl font-extrabold tracking-tight text-white sm:text-5xl md:text-6xl">
              Revolutionizing Cardiac Health with Strain Mapping
            </h1>
            <p className="mt-6 max-w-2xl mx-auto text-xl text-gray-300">
              Unsupervised Deep Learning for Early Detection and Myocardial
              Analysis
            </p>
            <div className="mt-10 flex justify-center gap-4">
              <button
                onClick={() => navigate("/viewer")}
                className="px-8 py-3 border border-transparent min-w-[220px] text-base font-medium rounded-md text-white bg-blue-600 shadow-none hover:shadow-lg transition-all duration-300 hover:bg-blue-700 md:py-4 md:text-lg md:px-10"
              >
                Try the Viewer
              </button>
              <button
                onClick={handleFeaturesClick}
                className="px-8 py-3 border border-white min-w-[220px] text-base font-medium rounded-md text-white bg-transparent hover:bg-white hover:text-gray-900 shadow-none hover:shadow-lg transition-all duration-300 ease-in-out md:py-4 md:text-lg md:px-10"
              >
                Explore Features
              </button>
            </div>
          </div>
        </div>
      </header>
    </>
  );
};

export default Header;
