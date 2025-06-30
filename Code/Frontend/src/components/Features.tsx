import { Heart, Monitor, Target, MessageSquare } from "lucide-react";

const Features = () => {
  return (
    <section id="features" className="py-20 bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h2 className="text-3xl font-extrabold text-center text-gray-900 mb-16">
          Key Features
        </h2>

        <div className="grid grid-cols-1 gap-8 md:grid-cols-2 lg:grid-cols-4">
          {[
            {
              icon: <Monitor className="h-8 w-8 text-blue-500" />,
              title: "Advanced DICOM Viewer",
              description:
                "Load and visualize complete patient studies with seamless cine MRI data integration.",
            },
            {
              icon: <Heart className="h-8 w-8 text-blue-500" />,
              title: "Myocardium Strain Analysis",
              description:
                "Analyze strain maps of the myocardium with high accuracy for early cardiac dysfunction detection.",
            },
            {
              icon: <Target className="h-8 w-8 text-blue-500" />,
              title: "Heart Localization",
              description:
                "Automatically localize the heart across entire series for precise analysis.",
            },
            {
              icon: <MessageSquare className="h-8 w-8 text-blue-500" />,
              title: "AI-Powered Chatbot",
              description:
                "Provides real-time guidance and insights for clinicians during analysis.",
            },
          ].map((feature, index) => (
            <div
              key={index}
              className="bg-white p-6 rounded-lg shadow-lg hover:shadow-2xl transition-shadow duration-300"
            >
              <div className="flex items-center justify-center w-12 h-12 rounded-md bg-blue-100 mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                {feature.title}
              </h3>
              <p className="text-gray-600">{feature.description}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Features;
