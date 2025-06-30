import React from "react";
import ChatBot from "react-chatbotify";

const Chatbot: React.FC = () => {
  const settings = {
    general: {
      primaryColor: "#2563eb", // Matches bg-blue-600
      fontFamily: "Inter, sans-serif",
      showHeader: true,
      showFooter: false,
      showInputRow: true,
      embedded: false,
    },
    chatHistory: {
      storageKey: "cardio_chat_history",
    },
    botBubble: {
      avatar: "src/assets/Cine-CMR-Img.png",
      showAvatar: true,
      animate: true,
    },
    header: {
      title: "CardioBot",
    },
    tooltip: {
      mode: "NEVER",
      text: "",
    },
  };

  const styles = {
    headerStyle: {
      backgroundColor: "#2563eb", // Matches bg-blue-600
      color: "#ffffff",
      fontSize: "0.9rem",
    },
    botMessageBubbleStyle: {
      backgroundColor: "#f3f4f6", // Matches bg-gray-100
      color: "#1f2937",
      fontSize: "0.85rem",
    },
    userMessageBubbleStyle: {
      backgroundColor: "#2563eb",
      color: "#ffffff",
      fontSize: "0.85rem",
    },
    chatButtonStyle: {
      width: "50px",
      height: "50px",
      backgroundColor: "#2563eb",
      borderRadius: "50%",
      padding: "8px",
    },
    chatWindowStyle: {
      fontSize: "0.80rem",
    },
    inputStyle: {
      fontSize: "0.85rem",
    },
  };

  const flow = {
    start: {
      message:
        "Welcome to CardioBot! I’m here to guide you through our DICOM viewer app. Our key features are: Advanced DICOM Viewer, Myocardium Strain Analysis, Heart Localization, and this AI-Powered Chatbot. Want to learn about a feature or ask about cardiac imaging?",
      transition: { duration: 0 },
      path: "main_menu",
    },
    main_menu: {
      message:
        "Select an option or type your question about cardiac imaging, strain mapping, or heart health:",
      options: [
        "Learn about features",
        "Explain DICOM Viewer",
        "Explain Strain Analysis",
        "Explain Heart Localization",
        "Explain Chatbot",
        "Ask about cardiac imaging",
        "Ask about heart disorders",
        "Ask about strain mapping",
        "Ask about other methods",
      ],
      path: (params: any) => {
        switch (params.userInput) {
          case "Learn about features":
            return "feature_list";
          case "Explain DICOM Viewer":
            return "dicom_viewer";
          case "Explain Strain Analysis":
            return "strain_analysis";
          case "Explain Heart Localization":
            return "heart_localization";
          case "Explain Chatbot":
            return "chatbot_info";
          case "Ask about cardiac imaging":
            return "cardiac_imaging";
          case "Ask about heart disorders":
            return "heart_disorders";
          case "Ask about strain mapping":
            return "strain_mapping";
          case "Ask about other methods":
            return "alternative_methods";
          default:
            return "handle_query";
        }
      },
    },
    feature_list: {
      message:
        "Our app offers: 1) Advanced DICOM Viewer to load patient studies, 2) Myocardium Strain Analysis for cardiac health, 3) Heart Localization for precise series analysis, and 4) this Chatbot for guidance. Pick one to explore!",
      path: "main_menu",
    },
    dicom_viewer: {
      message:
        "The Advanced DICOM Viewer lets you load entire patient studies. Select a series from the sidebar, scroll through images with your mouse wheel, and toggle metadata to view details like Series Number or Slice Thickness. Want to know more about navigation or tools?",
      path: "main_menu",
    },
    strain_analysis: {
      message:
        "Myocardium Strain Analysis uses cine MRI to measure heart muscle deformation, revealing regional health. It’s highly accurate for early detection of dysfunction. Curious about how it works or why it’s useful?",
      path: "main_menu",
    },
    heart_localization: {
      message:
        "Heart Localization automatically identifies the heart in a series using our localization model, enabling precise further analysis. Interested in the tech behind it?",
      path: "main_menu",
    },
    chatbot_info: {
      message:
        "I’m CardioBot, your AI guide! I explain features, answer questions on cardiac imaging, heart disorders, and strain mapping, and provide clinician support. Ask me anything about the app or heart health!",
      path: "main_menu",
    },
    cardiac_imaging: {
      message:
        "Cardiac imaging includes cine MRI, which captures heart motion over time, and CT for detailed structures. Cine MRI is key for strain analysis due to its high resolution. Want specifics on cine MRI or other modalities?",
      path: "main_menu",
    },
    heart_disorders: {
      message:
        "Common heart disorders include cardiomyopathy, heart failure, and arrhythmias. Strain analysis helps assess them by detecting abnormal muscle motion early. Need details on assessing disorders or specific conditions?",
      path: "main_menu",
    },
    strain_mapping: {
      message:
        "Strain mapping measures myocardial deformation using cine MRI and deep learning . It’s useful for early diagnosis, tracking disease progression, and guiding treatment. Want to dive into supervised vs. unsupervised methods?",
      path: "main_menu",
    },
    alternative_methods: {
      message:
        "Alternatives to strain mapping include speckle tracking (ultrasound), MR tagging, and ejection fraction (EF). Strain mapping excels in sensitivity and regional detail. Curious about how they compare?",
      path: "main_menu",
    },
    handle_query: {
      message: (params: any) => {
        const input = params.userInput.toLowerCase();
        if (input.includes("cine mri")) {
          return "Cine MRI captures heart motion in real-time, ideal for strain analysis due to its high temporal resolution. Want more on its role in cardiac imaging?";
        } else if (
          input.includes("heart function") ||
          input.includes("physiology")
        ) {
          return "The heart pumps blood via systole (contraction) and diastole (relaxation). Strain analysis assesses regional function for better diagnostics. Need specifics on disorders?";
        } else if (input.includes("assess disorders")) {
          return "Disorders are assessed using imaging (cine MRI, CT), biomarkers, and ECG. Strain mapping adds precision by measuring muscle strain. Want to explore one method?";
        } else if (
          input.includes("supervised") ||
          input.includes("unsupervised")
        ) {
          return "Supervised methods (e.g., U-Net) use labeled data for strain mapping, while unsupervised ones (e.g., VoxelMorph) learn from raw images, improving flexibility. More on one approach?";
        } else {
          return "I can help with cardiac imaging, strain mapping, heart disorders, or app features. Could you clarify or ask something specific?";
        }
      },
      path: "main_menu",
    },
  };

  return <ChatBot settings={settings} styles={styles} flow={flow} />;
};

export default Chatbot;
