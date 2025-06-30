import { FolderOpen } from "lucide-react";

function Visualization({ images, currentIndex, title, isLoading }: any) {
  const currentImage = images[currentIndex];

  return (
    <div className="flex flex-col">
      <h2 className="text-lg text-blue-600 font-semibold mb-2">{title}</h2>
      <div
        className={`relative ${
          title == "Series Visualization"
            ? "w-[600px] h-[600px]"
            : "w-[220px] h-[220px]"
        }
          border border-gray-300 rounded-lg overflow-hidden bg-gray-900`}
      >
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : images.length > 0 ? (
          <div className="relative w-full h-full">
            <img
              src={currentImage?.url}
              alt={`${title} - Frame ${currentIndex + 1}`}
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-2 right-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-sm">
              {currentIndex + 1} / {images.length}
            </div>
          </div>
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-400 gap-2">
            <FolderOpen className="w-12 h-12 opacity-50" />
            <span>No files loaded</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default Visualization;
