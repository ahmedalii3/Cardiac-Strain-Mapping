import {
  Move, // Pan
  ZoomIn, // Zoom
  Contrast, // Wwwc
  Ruler, // Length
  Square, // Rectangle
  Circle, // Ellipse
  CornerDownLeft, // Angle
  Paintbrush, // Brush
} from "lucide-react";

const DicomTools = ({ activateTool }: any) => {
  const tools = [
    { name: "Pan", icon: Move, tooltip: "Pan Tool" },
    { name: "Zoom", icon: ZoomIn, tooltip: "Zoom Tool" },
    { name: "Wwwc", icon: Contrast, tooltip: "Window/Level" },
    { name: "Length", icon: Ruler, tooltip: "Length Measurement" },
    { name: "RectangleRoi", icon: Square, tooltip: "Rectangle ROI" },
    { name: "EllipticalRoi", icon: Circle, tooltip: "Ellipse ROI" },
    { name: "Angle", icon: CornerDownLeft, tooltip: "Angle Measurement" },
    { name: "Brush", icon: Paintbrush, tooltip: "Brush Tool" },
  ];

  return (
    <>
      {tools.map((tool) => (
        <button
          key={tool.name}
          onClick={() => activateTool(tool.name)}
          className="p-2 text-gray-400 hover:text-white hover:bg-gray-700 rounded-md transition-colors relative group"
        >
          <tool.icon className="w-5 h-5" />
          <span className="absolute left-full ml-2 px-2 py-1 bg-gray-800 text-white text-xs rounded-md opacity-0 group-hover:opacity-100 whitespace-nowrap z-10">
            {tool.tooltip}
          </span>
        </button>
      ))}
    </>
  );
};

export default DicomTools;
