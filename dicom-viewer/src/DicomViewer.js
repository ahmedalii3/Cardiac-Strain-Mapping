import React, { useEffect, useRef } from "react";
import cornerstone from "cornerstone-core";
import dicomParser from "dicom-parser";
import cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";

cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;

cornerstoneWADOImageLoader.configure({
  beforeSend: function (xhr) {},
});

const DicomViewer = ({ file }) => {
  const elementRef = useRef(null);

  useEffect(() => {
    const element = elementRef.current;
    cornerstone.enable(element);

    const loadImageFromFile = (file) => {
      const fileReader = new FileReader();

      fileReader.onload = function (event) {
        const arrayBuffer = event.target.result;
        const imageId =
          cornerstoneWADOImageLoader.wadouri.fileManager.add(file);

        cornerstone
          .loadImage(imageId)
          .then((image) => {
            cornerstone.displayImage(element, image);
          })
          .catch((error) => {
            console.error("Error loading DICOM image:", error);
          });
      };

      fileReader.readAsArrayBuffer(file);
    };

    if (file) {
      loadImageFromFile(file);
    }

    return () => {
      cornerstone.disable(element);
    };
  }, [file]);

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-200">
      <div
        className="dicom-viewer border border-gray-400 bg-white"
        ref={elementRef}
        style={{ width: "512px", height: "512px" }}
      ></div>
    </div>
  );
};

export default DicomViewer;
