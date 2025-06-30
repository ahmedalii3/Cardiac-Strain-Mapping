import React, { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
// @ts-ignore
import * as cornerstone from "cornerstone-core";
// @ts-ignore
import * as cornerstoneMath from "cornerstone-math";
// @ts-ignore
import * as cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";
import * as dicomParser from "dicom-parser";
// @ts-ignore
import * as cornerstoneTools from "cornerstone-tools";
// @ts-ignore
import Hammer from "hammerjs";
import html2canvas from "html2canvas";
import DicomMetadata from "./DicomMetadata";
import DicomTools from "./DicomTools";
// @ts-ignore
import JSZip from "jszip";
// @ts-ignore
import { saveAs } from "file-saver";
import {
  CameraIcon,
  EyeIcon,
  EyeOffIcon,
  Loader2Icon,
  LocateIcon,
  UploadIcon,
  BringToFrontIcon,
} from "lucide-react";

cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
cornerstoneTools.external.cornerstoneMath = cornerstoneMath;

cornerstoneTools.init();

interface Series {
  studyID: string;
  seriesNumber: string;
  seriesDescription: string;
  modality: string;
  instances: { imageId: string; instanceNumber?: number }[];
  thumbnail: string | null;
}

interface Stack {
  imageIds: string[];
  currentImageIndex: number;
}

interface BinaryFile {
  name: string;
  type: string;
  binary: ArrayBuffer;
}

// interface SeriesFile {
//   filename: string;
//   content: string;
// }

interface Position {
  x: number;
  y: number;
}

const DicomViewer: React.FC = () => {
  const navigate = useNavigate();
  const dicomImageRef = useRef<HTMLDivElement>(null);
  const lastScrollTime = useRef<number>(0);
  const [groupedSeries, setGroupedSeries] = useState<Record<string, Series>>(
    {}
  );
  const [stack, setStack] = useState<Stack>({
    imageIds: [],
    currentImageIndex: 0,
  });
  const [selectedSeriesKey, setSelectedSeriesKey] = useState<string | null>(
    null
  );
  const [binaryDataGroups, setBinaryDataGroups] = useState<
    Record<string, BinaryFile[]>
  >({});
  const [processingLocalization, setProcessingLocalization] =
    useState<boolean>(false);
  const [processingStrain, setProcessingStrain] = useState<boolean>(false);
  const [showMetadata, setShowMetadata] = useState<boolean>(true);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [loadedImages, setLoadedImages] = useState<Record<string, any>>({});
  const [metadataList, setMetadataList] = useState<any[]>([]);
  const [currentMetadata, setCurrentMetadata] = useState<any>(null);
  const [mousePosition, setMousePosition] = useState<Position>({ x: 0, y: 0 });
  // @ts-ignore
  const [colormap, setColormap] = useState<string>("grayscale");

  useEffect(() => {
    if (!dicomImageRef.current) return;

    cornerstone.enable(dicomImageRef.current);

    const tools = [
      cornerstoneTools.PanTool,
      cornerstoneTools.ZoomTool,
      cornerstoneTools.WwwcTool,
      cornerstoneTools.LengthTool,
      cornerstoneTools.RectangleRoiTool,
      cornerstoneTools.EllipticalRoiTool,
      cornerstoneTools.AngleTool,
      cornerstoneTools.StackScrollMouseWheelTool,
      cornerstoneTools.BrushTool,
    ];

    tools.forEach((tool) =>
      cornerstoneTools.addToolForElement(dicomImageRef.current!, tool)
    );

    cornerstoneTools.setToolActiveForElement(dicomImageRef.current, "Pan", {
      mouseButtonMask: 1,
    });
    cornerstoneTools.setToolActiveForElement(
      dicomImageRef.current,
      "StackScrollMouseWheel",
      {}
    );

    return () => {
      cornerstone.disable(dicomImageRef.current!);
    };
  }, []);

  useEffect(() => {
    if (stack.imageIds.length > 0) {
      loadDicomImage(stack.imageIds[stack.currentImageIndex]);
    }
  }, [stack]);

  useEffect(() => {
    if (
      stack.imageIds.length > 0 &&
      dicomImageRef.current &&
      loadedImages[stack.imageIds[stack.currentImageIndex]]
    ) {
      applyColormap(loadedImages[stack.imageIds[stack.currentImageIndex]]);
    }
  }, [colormap, stack.currentImageIndex]);

  const applyColormap = (image: any) => {
    const element = dicomImageRef.current;
    if (!element || !image) return;

    const viewport = cornerstone.getViewport(element);
    if (!viewport) return;

    if (colormap === "grayscale") {
      viewport.colormap = undefined; // Reset to default grayscale
    } else {
      viewport.colormap = colormap; // Use Cornerstone's built-in colormaps
    }

    cornerstone.setViewport(element, viewport);
    cornerstone.updateImage(element);
  };

  const handleFolderUpload = async (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setIsLoading(true);
    const files = Array.from(event.target.files || []).filter((file) =>
      file.name.endsWith(".dcm")
    );
    if (files.length > 0) {
      await processDicomFiles(files);
    }
    setIsLoading(false);
  };

  const processDicomFiles = async (files: File[]) => {
    const grouped: Record<string, Series> = {};
    const binaryGroups: Record<string, BinaryFile[]> = {};

    // Temporary array to store file metadata for sorting
    const fileMetadata: Array<{
      file: File;
      arrayBuffer: ArrayBuffer;
      dataset: any;
      studyID: string;
      seriesNumber: string;
      instanceNumber: number;
      fileSequenceNumber: number;
    }> = [];

    // Function to extract sequence number from filename
    const extractSequenceNumber = (filename: string): number => {
      // Match numbers in filenames like 'slice_007 2.dcm', 'slice_007-2.dcm', 'IM-0003-0179.dcm', or 'slice_007.dcm'
      const match = filename.match(/(\d+)(?=\.dcm$)|(\d+)(?=\s|$|-)/i);
      if (match) {
        return parseInt(match[1] || match[2] || "0");
      }
      // For files like 'slice_007.dcm' with no explicit sequence number, assume 1
      return filename.includes(".dcm") ? 1 : 0;
    };

    // Read metadata and collect instance numbers
    for (const file of files) {
      const arrayBuffer = await file.arrayBuffer();
      const dataset = dicomParser.parseDicom(new Uint8Array(arrayBuffer));

      const studyID = dataset.string("x00200010") || "Unknown Study";
      const seriesNumber = dataset.string("x00200011") || "Unknown Series";
      const instanceNumber = parseInt(dataset.string("x00200013") || "0");
      const fileSequenceNumber = extractSequenceNumber(file.name);

      fileMetadata.push({
        file,
        arrayBuffer,
        dataset,
        studyID,
        seriesNumber,
        instanceNumber,
        fileSequenceNumber,
      });
    }

    // Sort files by instanceNumber first, then by fileSequenceNumber
    fileMetadata.sort((a, b) => {
      // Use instanceNumber if both are valid and non-zero
      if (
        a.instanceNumber > 0 &&
        b.instanceNumber > 0 &&
        a.instanceNumber !== b.instanceNumber
      ) {
        return a.instanceNumber - b.instanceNumber;
      }
      // Fallback to fileSequenceNumber
      return a.fileSequenceNumber - b.fileSequenceNumber;
    });

    // Process sorted files
    for (const {
      file,
      arrayBuffer,
      dataset,
      studyID,
      seriesNumber,
      instanceNumber,
    } of fileMetadata) {
      const seriesKey = `${studyID}|${seriesNumber}`;
      const seriesDescription =
        dataset.string("x0008103e") || "Unknown Description";
      const modality = dataset.string("x00080060") || "Unknown Modality";

      if (!grouped[seriesKey]) {
        grouped[seriesKey] = {
          studyID,
          seriesNumber,
          seriesDescription,
          modality,
          instances: [],
          thumbnail: null,
        };
        binaryGroups[seriesKey] = [];
      }

      const fileUrl = URL.createObjectURL(file);
      const imageId = `wadouri:${fileUrl}`;

      grouped[seriesKey].instances.push({
        imageId,
        instanceNumber: dataset.string("x00200013"),
      });
      binaryGroups[seriesKey].push({
        name: file.name,
        type: file.type,
        binary: arrayBuffer,
      });

      if (!grouped[seriesKey].thumbnail) {
        try {
          const image = await cornerstone.loadImage(imageId);
          const canvas = document.createElement("canvas");
          canvas.width = 100;
          canvas.height = 100;
          const context = canvas.getContext("2d")!;
          context.fillStyle = "black";
          context.fillRect(0, 0, 100, 100);
          cornerstone.renderToCanvas(canvas, image);
          grouped[seriesKey].thumbnail = canvas.toDataURL();
        } catch (error) {
          console.error("Error generating thumbnail:", error);
        }
      }
    }

    setGroupedSeries(grouped);
    setBinaryDataGroups(binaryGroups);
    console.log("grouped", grouped);
    console.log("binary", binaryGroups);
  };

  const handleMouseMove = useCallback(
    (event: React.MouseEvent<HTMLDivElement>) => {
      if (!dicomImageRef.current || !currentMetadata) return;

      const element = dicomImageRef.current;
      const rect = element.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const resolution = currentMetadata.Resolution || [512, 512];
      const scaleX = resolution[0] / element.offsetWidth;
      const scaleY = resolution[1] / element.offsetHeight;
      const imageX = Math.floor(x * scaleX);
      const imageY = Math.floor(y * scaleY);

      setMousePosition({ x: imageX, y: imageY });
    },
    [currentMetadata]
  );

  const handleSeriesClick = async (series: Series) => {
    setIsLoading(true);
    if (dicomImageRef.current) {
      cornerstoneTools.clearToolState(dicomImageRef.current, "Brush");
      cornerstoneTools.clearToolState(dicomImageRef.current, "stack");
    }
    console.log("series", series);
    const seriesKey = `${series.studyID}|${series.seriesNumber}`;

    // Function to extract sequence number from imageId (derived from filename)
    const extractSequenceNumber = (imageId: string): number => {
      const filename = imageId.split("/").pop() || "";
      const match = filename.match(/(\d+)(?=\.dcm$)|(\d+)(?=\s|$|-)/i);
      if (match) {
        return parseInt(match[1] || match[2] || "0");
      }
      return filename.includes(".dcm") ? 1 : 0;
    };

    // Sort instances by instanceNumber first, then by fileSequenceNumber
    const sortedInstances = series.instances.sort((a, b) => {
      const aNum = parseInt(a.instanceNumber?.toString() || "0");
      const bNum = parseInt(b.instanceNumber?.toString() || "0");
      if (aNum > 0 && bNum > 0 && aNum !== bNum) {
        return aNum - bNum;
      }
      return (
        extractSequenceNumber(a.imageId) - extractSequenceNumber(b.imageId)
      );
    });

    console.log("sorted", sortedInstances);
    const imageIds = sortedInstances.map((instance) => instance.imageId);
    setStack({ imageIds, currentImageIndex: 0 });
    setSelectedSeriesKey(seriesKey);

    if (binaryDataGroups[seriesKey]) {
      const formData = new FormData();
      const filesToProcess = binaryDataGroups[seriesKey];
      filesToProcess.forEach((file) => {
        const blob = new Blob([file.binary], { type: file.type });
        formData.append("files", blob, file.name);
      });

      try {
        const metadataResponse = await fetch("http://127.0.0.1:8000/metadata", {
          method: "POST",
          body: formData,
        });
        if (metadataResponse.ok) {
          const data = await metadataResponse.json();
          const sortedMetadata = data.metadata.sort((a: any, b: any) => {
            const aNum = parseInt(a.InstanceNumber) || 0;
            const bNum = parseInt(b.InstanceNumber) || 0;
            if (aNum > 0 && bNum > 0 && aNum !== bNum) {
              return aNum - bNum;
            }
            // Fallback to filename-based sorting for metadata
            const aFileNum = extractSequenceNumber(a.filename || "");
            const bFileNum = extractSequenceNumber(b.filename || "");
            return aFileNum - bFileNum;
          });
          setMetadataList(sortedMetadata);
          setCurrentMetadata(sortedMetadata[0]);
        }
      } catch (error) {
        console.error("Error fetching series metadata:", error);
      }
    }
    setIsLoading(false);
  };
  const handleCalculateStrain = async () => {
    if (!selectedSeriesKey || !binaryDataGroups[selectedSeriesKey]) {
      alert("No series selected.");
      return;
    }

    setProcessingStrain(true);
    setIsLoading(true);

    try {
      const arrayBufferToBase64 = (buffer: ArrayBuffer) => {
        const binary = new Uint8Array(buffer);
        let binaryString = "";
        for (let i = 0; i < binary.length; i += 32768) {
          binaryString += String.fromCharCode(...binary.subarray(i, i + 32768));
        }
        return btoa(binaryString);
      };

      const formData = new FormData();
      binaryDataGroups[selectedSeriesKey].forEach((file) => {
        const blob = new Blob([file.binary], { type: file.type });
        formData.append("files", blob, file.name);
      });

      const response = await fetch("http://127.0.0.1:8000/strain/dicom", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Strain calculation response:", data);
        console.log("Strain calculation metadata:", data.metadata);
        const localizedSeriesData = data.localized.map((file: any) => ({
          filename: file.filename,
          content: file.content,
        }));
        const strain1Data = data.strain1.map((file: any) => ({
          filename: file.filename,
          content: file.content,
          strain_min: file.strain_min, // Pass strain_min
          strain_max: file.strain_max, // Pass strain_max
        }));
        const strain2Data = data.strain2.map((file: any) => ({
          filename: file.filename,
          content: file.content,
          strain_min: file.strain_min,
          strain_max: file.strain_max,
        }));
        const strain3Data = data.strain3.map((file: any) => ({
          filename: file.filename,
          content: file.content,
          strain_min: file.strain_min,
          strain_max: file.strain_max,
        }));

        navigate("/results", {
          state: {
            originalSeries: binaryDataGroups[selectedSeriesKey].map((file) => ({
              filename: file.name,
              content: arrayBufferToBase64(file.binary),
            })),
            localizedSeries: localizedSeriesData,
            strain1: strain1Data,
            strain2: strain2Data,
            strain3: strain3Data,
            metadata: metadataList,
            masks: data.masks || [],
            bullseye1: data.bullseye1 || null,
            bullseye2: data.bullseye2 || null,
            bullseye3: data.bullseye3 || null,
            bullseye1Ring: data.bullseye1_ring || null,
            bullseye2Ring: data.bullseye2_ring || null,
            bullseye3Ring: data.bullseye3_ring || null,
            segmentMeans: data.segment_means || [],
          },
        });
      } else {
        console.error("Strain calculation failed:", await response.text());
        alert("Failed to calculate strain.");
      }
    } catch (error) {
      console.error("Error during strain calculation:", error);
      alert("Failed to navigate to results due to an error.");
    }

    setProcessingStrain(false);
    setIsLoading(false);
  };

  const loadDicomImage = (imageId: string) => {
    const element = dicomImageRef.current;
    if (!element) return;

    if (loadedImages[imageId]) {
      cornerstone.displayImage(element, loadedImages[imageId]);
      setViewportForImage(loadedImages[imageId]);
      applyColormap(loadedImages[imageId]);
    } else {
      setIsLoading(true);
      cornerstone.loadImage(imageId).then((image: any) => {
        setLoadedImages((prev) => ({ ...prev, [imageId]: image }));
        cornerstone.displayImage(element, image);
        setViewportForImage(image);
        applyColormap(image);
        setIsLoading(false);
      });
    }

    const stackState = { currentImageIdIndex: 0, imageIds: [imageId] };
    cornerstoneTools.addStackStateManager(element, ["stack", "brush"]);
    cornerstoneTools.addToolState(element, "stack", stackState);
  };

  const setViewportForImage = (image: any) => {
    const element = dicomImageRef.current;
    if (!element) return;
    const defaultViewport = cornerstone.getDefaultViewportForImage(
      element,
      image
    );
    cornerstone.setViewport(element, defaultViewport);
  };

  const handleScroll = useCallback(
    (event: WheelEvent) => {
      event.preventDefault();
      const now = Date.now();
      if (now - lastScrollTime.current < 200) return;
      lastScrollTime.current = now;

      const direction = Math.sign(event.deltaY);
      const nextIndex = Math.min(
        Math.max(0, stack.currentImageIndex + direction),
        stack.imageIds.length - 1
      );

      if (nextIndex !== stack.currentImageIndex) {
        setStack((prev) => ({ ...prev, currentImageIndex: nextIndex }));
      }
    },
    [stack]
  );

  useEffect(() => {
    const element = dicomImageRef.current;
    if (!element) return;

    element.addEventListener("wheel", handleScroll, { passive: false });
    return () => element.removeEventListener("wheel", handleScroll);
  }, [handleScroll]);

  useEffect(() => {
    if (stack.currentImageIndex < metadataList.length) {
      setCurrentMetadata(metadataList[stack.currentImageIndex]);
    }
  }, [stack.currentImageIndex, metadataList]);

  const activateTool = (toolName: string) => {
    const element = dicomImageRef.current;
    if (!element) return;
    cornerstoneTools.setToolActiveForElement(element, toolName, {
      mouseButtonMask: 1,
    });
  };

  const exportToPNG = () => {
    const element = dicomImageRef.current;
    if (!element) return;

    html2canvas(element, { useCORS: true, logging: false })
      .then((canvas) => {
        const dataURL = canvas.toDataURL("image/png");
        const link = document.createElement("a");
        link.href = dataURL;
        link.download = "dicom_with_annotations.png";
        link.click();
      })
      .catch((error) => console.error("Error capturing screenshot:", error));
  };

  const handleLocalization = async () => {
    if (!selectedSeriesKey || !binaryDataGroups[selectedSeriesKey]) {
      console.error("No series selected");
      return;
    }

    setProcessingLocalization(true);
    const seriesData = binaryDataGroups[selectedSeriesKey];
    const formData = new FormData();

    seriesData.forEach((file) => {
      const blob = new Blob([file.binary], { type: file.type });
      formData.append("files", blob, file.name);
    });

    try {
      const response = await fetch("http://127.0.0.1:8000/localize", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error("Localization failed");

      const data = await response.json();
      const instances = await Promise.all(
        data.files.map(async (file: any, index: number) => ({
          imageId: `wadouri:data:application/dicom;base64,${file.content}`,
          instanceNumber: index + 1,
        }))
      );

      const thumbnail = await generateThumbnailFromBase64(
        data.files[0].content
      );
      const localizedSeriesKey = `${data.series_uid}|Localized`;
      const localizedSeries: Series = {
        studyID: data.series_uid,
        seriesNumber: "Localized",
        seriesDescription:
          data.series_description || "Localized CINESAX Series",
        modality: data.modality || "CINESAX",
        instances,
        thumbnail,
      };

      const localizedBinaryGroup = data.files.map((file: any) => ({
        name: file.filename,
        type: "application/dicom",
        binary: Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0))
          .buffer,
      }));

      setGroupedSeries((prev) => ({
        ...prev,
        [localizedSeriesKey]: localizedSeries,
      }));
      setBinaryDataGroups((prev) => ({
        ...prev,
        [localizedSeriesKey]: localizedBinaryGroup,
      }));
      await handleSeriesClick(localizedSeries);
    } catch (error) {
      console.error("Error during localization:", error);
    } finally {
      setProcessingLocalization(false);
    }
  };

  const generateThumbnailFromBase64 = async (
    base64Content: string
  ): Promise<string | null> => {
    try {
      const binaryString = atob(base64Content);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }

      const blob = new Blob([bytes], { type: "application/dicom" });
      const imageId = URL.createObjectURL(blob);
      const image = await cornerstone.loadImage(`wadouri:${imageId}`);

      const canvas = document.createElement("canvas");
      canvas.width = 100;
      canvas.height = 100;
      const context = canvas.getContext("2d")!;
      context.fillStyle = "black";
      context.fillRect(0, 0, 100, 100);
      await cornerstone.renderToCanvas(canvas, image);

      URL.revokeObjectURL(imageId);
      return canvas.toDataURL();
    } catch (error) {
      console.error("Error generating thumbnail:", error);
      return null;
    }
  };

  return (
    <div className="flex flex-1 pt-16 bg-gray-900 text-gray-100">
      {/* Sidebar - Series List */}
      <div
        className="w-64 bg-gray-800 p-4 overflow-y-auto h-full border-r border-gray-700"
        style={{ height: "calc(100vh + 4rem)" }}
      >
        <h2 className="font-bold text-lg mb-4 text-blue-300">Series List</h2>
        {Object.keys(groupedSeries).map((seriesKey) => {
          const series = groupedSeries[seriesKey];
          return (
            <div
              key={seriesKey}
              className={`p-3 mb-2 rounded-md cursor-pointer flex items-start gap-3 transition-colors ${
                selectedSeriesKey === seriesKey
                  ? "bg-blue-900/50 border border-blue-500"
                  : "hover:bg-gray-700 border border-gray-700"
              }`}
              onClick={() => handleSeriesClick(series)}
            >
              {series.thumbnail && (
                <img
                  src={series.thumbnail}
                  alt={`Series ${series.seriesNumber}`}
                  className="w-16 h-16 border border-gray-600 rounded-sm object-contain bg-black"
                />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium truncate">
                  {series.modality || "Unknown"} - {series.seriesDescription}
                </p>
                <p className="text-xs text-gray-400">
                  {series.instances.length} images
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Main Content Area */}
      <div className={`flex-1 flex flex-col ${isLoading ? "opacity-50" : ""}`}>
        {/* Top Toolbar */}
        <div className="bg-gray-800 p-3 border-b border-gray-700 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <label
              htmlFor="file-upload"
              className="cursor-pointer bg-blue-600 hover:bg-blue-700 text-white px-3 py-1.5 rounded-md text-sm flex items-center gap-1 transition-colors"
            >
              <UploadIcon className="w-4 h-4" />
              <span>Upload Folder</span>
            </label>
            <input
              id="file-upload"
              type="file"
              multiple
              webkitdirectory="true"
              directory="true"
              className="hidden"
              onChange={handleFolderUpload}
            />

            <button
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm flex items-center gap-1 transition-colors"
              onClick={exportToPNG}
            >
              <CameraIcon className="w-4 h-4" />
              <span>Export PNG</span>
            </button>
          </div>
          {/* Processing Buttons */}
          <div className="bg-gray-800 p-3 border-b border-gray-700 flex items-center justify-center gap-3">
            <button
              className={`px-3 py-1.5 rounded-md text-sm flex items-center gap-1 transition-colors ${
                processingLocalization
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white"
              }`}
              onClick={handleLocalization}
              disabled={processingLocalization}
            >
              {processingLocalization ? (
                <Loader2Icon className="w-4 h-4 animate-spin" />
              ) : (
                <LocateIcon className="w-4 h-4" />
              )}
              <span>Localize Heart</span>
            </button>

            <button
              className={`px-3 py-1.5 rounded-md text-sm flex items-center gap-1 transition-colors ${
                processingStrain || !selectedSeriesKey
                  ? "bg-gray-600 cursor-not-allowed"
                  : "bg-blue-600 hover:bg-blue-700 text-white"
              }`}
              onClick={handleCalculateStrain}
              disabled={processingStrain || !selectedSeriesKey}
            >
              {processingStrain ? (
                <Loader2Icon className="w-4 h-4 animate-spin" />
              ) : (
                <BringToFrontIcon className="w-4 h-4" />
              )}
              <span>Calculate Strain</span>
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              className="px-3 py-1.5 bg-blue-600 hover:bg-blue-700 text-white rounded-md text-sm flex items-center gap-1 transition-colors"
              onClick={() => setShowMetadata(!showMetadata)}
            >
              {showMetadata ? (
                <>
                  <EyeOffIcon className="w-4 h-4" />
                  <span>Hide Metadata</span>
                </>
              ) : (
                <>
                  <EyeIcon className="w-4 h-4" />
                  <span>Show Metadata</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Viewer Area */}
        <div className="flex flex-1 overflow-hidden">
          {/* Tools Panel (Vertical) */}
          <div className="w-12 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4 gap-3">
            <DicomTools activateTool={activateTool} />
          </div>

          {/* Image Area */}
          <div className="flex-1 flex flex-col items-center justify-center p-4 bg-gray-900">
            <div
              ref={dicomImageRef}
              className="border border-gray-700 rounded-md relative bg-black"
              onMouseMove={handleMouseMove}
              style={{ width: "812px", height: "612px" }}
            >
              {stack.imageIds.length === 0 && (
                <p className="text-center text-gray-400 mt-20">
                  Select a series to view
                </p>
              )}
              {showMetadata && currentMetadata && (
                <DicomMetadata
                  metadata={currentMetadata}
                  position={mousePosition}
                />
              )}
            </div>
            {stack.imageIds.length > 0 && (
              <p className="text-center text-gray-400 py-2 text-sm">
                Image {stack.currentImageIndex + 1} of {stack.imageIds.length}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Loading Indicator */}
      {isLoading && (
        <div className="absolute inset-0 flex justify-center items-center bg-gray-900/80 z-50">
          <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500"></div>
        </div>
      )}
    </div>
  );
};

export default DicomViewer;
