import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";
import DicomViewer from "./DicomViewer";

function App() {
  const [file, setFile] = useState(null);

  const handleFileChange = (event) => {
    const uploadedFile = event.target.files[0];
    if (uploadedFile) {
      setFile(uploadedFile);
    }
  };

  return (
    <div className="App">
      <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
        <label className="mb-4">
          <input
            type="file"
            accept=".dcm"
            onChange={handleFileChange}
            className="hidden"
            id="file-upload"
          />
          <button
            className="px-4 py-2 bg-blue-500 text-white rounded"
            onClick={() => document.getElementById("file-upload").click()}
          >
            Upload DICOM File
          </button>
        </label>
        {file && <DicomViewer file={file} />}
      </div>
    </div>
  );
}

export default App;
