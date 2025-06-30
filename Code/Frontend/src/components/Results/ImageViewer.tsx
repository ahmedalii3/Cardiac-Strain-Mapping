import React, {
  useEffect,
  useRef,
  useState,
  useCallback,
  useMemo,
} from "react";
import { useLocation } from "react-router-dom";
// @ts-ignore
import * as cornerstone from "cornerstone-core";
// @ts-ignore
import * as cornerstoneMath from "cornerstone-math";
// @ts-ignore
import * as cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";
// @ts-ignore
import * as dicomParser from "dicom-parser";
// @ts-ignore
import * as cornerstoneTools from "cornerstone-tools";
// @ts-ignore
import Hammer from "hammerjs";
// @ts-ignore
import JSZip from "jszip";
// @ts-ignore
import { saveAs } from "file-saver";
import FrameBullseyeModal from "./FrameBullseyeModal";
import { Button } from "../ui/button";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Download,
  Info,
  FileText,
  Eye,
  EyeOff,
} from "lucide-react";

cornerstoneTools.external.cornerstone = cornerstone;
cornerstoneTools.external.Hammer = Hammer;
cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
cornerstoneTools.init();

cornerstoneWADOImageLoader.configure({
  useWebGL: true,
  webglDebug: true,
  interpolation: "bilinear",
});

interface SeriesFile {
  filename: string;
  content: string;
  strain_min?: number;
  strain_max?: number;
  histogram_bins?: number[];
  histogram_counts?: number[];
  percentile_5?: number;
  percentile_95?: number;
}

interface PlaybackState {
  isPlaying: boolean;
  currentIndex: number;
  speed: number;
}

interface Stack {
  imageIds: string[];
  currentImageIndex: number;
}

interface SegmentMeans {
  frame: number;
  segment_means: number[];
}

const ImageViewer: React.FC = () => {
  const location = useLocation();
  const {
    originalSeries,
    localizedSeries,
    strain1,
    strain2,
    strain3,
    metadata = [],
    masks = [],
    bullseye1 = [],
    bullseye2 = [],
    bullseye3 = [],
    bullseye1Ring = [],
    bullseye2Ring = [],
    bullseye3Ring = [],
    segmentMeans = { bullseye1: [], bullseye2: [], bullseye3: [] },
  } = location.state || {};

  const dicomImageRefs = useRef<(HTMLDivElement | null)[]>([
    null,
    null,
    null,
    null,
  ]);
  const colorBarRefs = useRef<(HTMLCanvasElement | null)[]>([
    null,
    null,
    null,
    null,
  ]);
  const [stacks, setStacks] = useState<{
    localized: Stack;
    strain1: Stack;
    strain2: Stack;
    strain3: Stack;
  }>({
    localized: { imageIds: [], currentImageIndex: 0 },
    strain1: { imageIds: [], currentImageIndex: 0 },
    strain2: { imageIds: [], currentImageIndex: 0 },
    strain3: { imageIds: [], currentImageIndex: 0 },
  });
  const [playbackState, setPlaybackState] = useState<PlaybackState>({
    isPlaying: false,
    currentIndex: 0,
    speed: 1,
  });
  const [isLoading, setIsLoading] = useState(false);
  const [colormaps, setColormaps] = useState<string[]>([
    "grayscale",
    "jet",
    "jet",
    "jet",
  ]);
  const [error, setError] = useState<string | null>(null);
  const [elementsEnabled, setElementsEnabled] = useState(false);
  const [strainRanges, setStrainRanges] = useState<
    { min: number; max: number }[]
  >([
    { min: 0, max: 1 },
    { min: -0.5, max: 0.5 },
    { min: -0.5, max: 0.5 },
    { min: -0.5, max: 0.5 },
  ]);
  const [isStrainRangesReady, setIsStrainRangesReady] = useState(false);
  const [hasInteractedWithWWWC, setHasInteractedWithWWWC] = useState<boolean[]>(
    [false, false, false, false]
  );
  const [justChangedFrame, setJustChangedFrame] = useState<boolean>(false);
  const [showMasks, setShowMasks] = useState<boolean[]>([
    false,
    true,
    true,
    true,
  ]);
  const [maskData, setMaskData] = useState<Map<number, number[][]>>(new Map());
  const [isBullseyeModalOpen, setIsBullseyeModalOpen] = useState(false);

  const globalStrainRanges = useMemo(
    () =>
      ({
        strain1: null,
        strain2: null,
        strain3: null,
      } as {
        strain1: { min: number; max: number } | null;
        strain2: { min: number; max: number } | null;
        strain3: { min: number; max: number } | null;
      }),
    []
  );

  const initialWindowCenters = useMemo(
    () => ({
      strain1: 10000,
      strain2: 43000,
      strain3: 23000,
    }),
    []
  );

  useEffect(() => {
    if (!localizedSeries || !strain1 || !strain2 || !strain3) {
      setError("Missing series or strain data");
      console.error("Missing data in location.state", {
        localizedSeries,
        strain1,
        strain2,
        strain3,
      });
      return;
    }

    const initializeStacks = async () => {
      setIsLoading(true);
      try {
        const computeStrainRange = (files: SeriesFile[]) => {
          const mins = files
            .map((file) => file.strain_min)
            .filter(
              (val): val is number => typeof val === "number" && isFinite(val)
            );
          const maxs = files
            .map((file) => file.strain_max)
            .filter(
              (val): val is number => typeof val === "number" && isFinite(val)
            );
          return {
            min: mins.length > 0 ? Math.min(...mins) : -0.5,
            max: maxs.length > 0 ? Math.max(...maxs) : 0.5,
          };
        };

        const validatedRanges = {
          strain1: computeStrainRange(strain1),
          strain2: computeStrainRange(strain2),
          strain3: computeStrainRange(strain3),
        };

        globalStrainRanges.strain1 = validatedRanges.strain1;
        globalStrainRanges.strain2 = validatedRanges.strain2;
        globalStrainRanges.strain3 = validatedRanges.strain3;

        setStrainRanges([
          { min: 0, max: 1 },
          validatedRanges.strain1,
          validatedRanges.strain2,
          validatedRanges.strain3,
        ]);
        setIsStrainRangesReady(true);

        const createImageIds = (series: SeriesFile[]) =>
          series.map(
            (file) => `wadouri:data:application/dicom;base64,${file.content}`
          );

        setStacks({
          localized: {
            imageIds: createImageIds(localizedSeries),
            currentImageIndex: 0,
          },
          strain1: { imageIds: createImageIds(strain1), currentImageIndex: 0 },
          strain2: { imageIds: createImageIds(strain2), currentImageIndex: 0 },
          strain3: { imageIds: createImageIds(strain3), currentImageIndex: 0 },
        });
      } catch (err) {
        console.error("Error initializing stacks:", err);
        setError("Failed to initialize series data");
      } finally {
        setIsLoading(false);
      }
    };

    initializeStacks();
  }, [localizedSeries, strain1, strain2, strain3, globalStrainRanges]);

  useEffect(() => {
    const zeroMask = new Array(128).fill(0).map(() => new Array(128).fill(0));
    const newMaskData = new Map<number, number[][]>();

    if (masks && masks.length > 0) {
      masks.forEach(
        (mask: { filename: string; values: number[][] }, index: number) => {
          if (
            mask.values &&
            mask.values.length === 128 &&
            mask.values[0].length === 128
          ) {
            console.log(
              `Loaded mask array for frame ${index + 1}, shape: ${
                mask.values.length
              }x${mask.values[0].length}`
            );
            newMaskData.set(index, mask.values);
          } else {
            console.warn(`Invalid mask data for frame ${index + 1}`);
            newMaskData.set(index, zeroMask);
          }
        }
      );
    } else {
      console.warn("No masks provided in response");
      Array.from({ length: stacks.localized.imageIds.length }).forEach(
        (_, index) => {
          newMaskData.set(index, zeroMask);
        }
      );
    }

    setMaskData(newMaskData);
  }, [masks, stacks.localized.imageIds.length]);

  useEffect(() => {
    const enableElements = () => {
      dicomImageRefs.current.forEach((element, index) => {
        if (!element) return;
        try {
          cornerstone.enable(element);
          cornerstoneTools.addTool(cornerstoneTools.WwwcTool);
          cornerstoneTools.addTool(cornerstoneTools.StackScrollMouseWheelTool);
          cornerstoneTools.setToolActive("Wwwc", { mouseButtonMask: 1 });
          cornerstoneTools.setToolActive("StackScrollMouseWheel", {});

          const handleImageRendered = (event: any) => {
            if (index === 0) return;
            const viewport = cornerstone.getViewport(element);
            if (!viewport) return;

            const currentWW = viewport.voi.windowWidth;
            const currentWC = viewport.voi.windowCenter;
            const initialWW = 65535;
            const initialWC =
              initialWindowCenters[
                ["strain1", "strain2", "strain3"][
                  index - 1
                ] as keyof typeof initialWindowCenters
              ];

            if (!justChangedFrame) {
              setHasInteractedWithWWWC((prev) => {
                const newState = [...prev];
                newState[index] =
                  Math.abs(currentWW - initialWW) > 1 ||
                  Math.abs(currentWC - initialWC) > 1;
                return newState;
              });
            } else {
              setJustChangedFrame(false);
            }
          };

          element.addEventListener(
            "cornerstoneimagerendered",
            handleImageRendered
          );
          element.__imageRenderedHandler = handleImageRendered;
        } catch (err) {
          console.error(
            `Error enabling Cornerstone for viewport ${index}:`,
            err
          );
          setError(`Failed to initialize viewport ${index}`);
        }
      });
      setElementsEnabled(true);
    };

    if (dicomImageRefs.current.every((el) => el !== null)) {
      enableElements();
    }

    return () => {
      dicomImageRefs.current.forEach((element, index) => {
        if (!element) return;
        try {
          if (element.__imageRenderedHandler) {
            element.removeEventListener(
              "cornerstoneimagerendered",
              element.__imageRenderedHandler
            );
          }
          cornerstone.disable(element);
        } catch (err) {
          console.error(
            `Error disabling Cornerstone for viewport ${index}:`,
            err
          );
        }
      });
      setElementsEnabled(false);
    };
  }, [initialWindowCenters]);

  const drawColorBar = useCallback(
    (
      canvas: HTMLCanvasElement | null,
      viewportIndex: number,
      strainData: SeriesFile[]
    ) => {
      if (
        !canvas ||
        colormaps[viewportIndex] === "grayscale" ||
        !isStrainRangesReady
      )
        return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const width = canvas.width;
      const height = canvas.height;
      const element = dicomImageRefs.current[viewportIndex];
      const viewport = element ? cornerstone.getViewport(element) : null;

      ctx.clearRect(0, 0, width, height);

      if (viewport && strainData.length > 0) {
        const windowWidth = viewport.voi.windowWidth;
        const initialRange = 1.0;
        const initialCenter = 0.0;
        const initialWW = 65535;

        const scaleFactor = windowWidth / initialWW;
        const newRange = Math.min(initialRange * scaleFactor, 2.0);
        const newMin = initialCenter - newRange / 2;
        const newMax = initialCenter + newRange / 2;

        if (hasInteractedWithWWWC[viewportIndex]) {
          setStrainRanges((prev) => {
            const currentRange = prev[viewportIndex];
            if (currentRange.min === newMin && currentRange.max === newMax) {
              return prev;
            }
            const newRanges = [...prev];
            newRanges[viewportIndex] = { min: newMin, max: newMax };
            return newRanges;
          });
        }

        const gradient = ctx.createLinearGradient(0, height, 0, 0);
        const colormap = colormaps[viewportIndex];
        if (colormap === "jet") {
          gradient.addColorStop(0, "blue");
          gradient.addColorStop(0.2, "cyan");
          gradient.addColorStop(0.5, "rgb(77, 213, 75)");
          gradient.addColorStop(0.7, "yellow");
          gradient.addColorStop(0.9, "red");
          gradient.addColorStop(1, "rgb(145, 9, 9)");
        } else if (colormap === "hot") {
          gradient.addColorStop(0, "black");
          gradient.addColorStop(0.33, "red");
          gradient.addColorStop(0.66, "yellow");
          gradient.addColorStop(1, "white");
        } else if (colormap === "cool") {
          gradient.addColorStop(0, "cyan");
          gradient.addColorStop(1, "magenta");
        }

        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
      }
    },
    [colormaps, isStrainRangesReady, hasInteractedWithWWWC]
  );

  const updateColorBar = useCallback(
    (viewportIndex: number) => {
      const canvas = colorBarRefs.current[viewportIndex];
      const strainData = [strain1, strain2, strain3][viewportIndex - 1];
      if (canvas && strainData && strainData.length > 0) {
        drawColorBar(canvas, viewportIndex, strainData);
      }
    },
    [strain1, strain2, strain3, drawColorBar]
  );

  useEffect(() => {
    colorBarRefs.current.forEach((canvas, index) => {
      if (index > 0 && isStrainRangesReady && canvas) {
        const viewer = dicomImageRefs.current[index];
        if (viewer) {
          canvas.height = viewer.getBoundingClientRect().height;
          drawColorBar(canvas, index, [strain1, strain2, strain3][index - 1]);
        }
      }
    });
  }, [
    colormaps,
    isStrainRangesReady,
    strainRanges,
    hasInteractedWithWWWC,
    drawColorBar,
  ]);

  const loadImageForViewport = useCallback(
    async (
      element: HTMLDivElement | null,
      stack: Stack,
      viewportIndex: number
    ) => {
      if (!element || stack.imageIds.length === 0) return;

      try {
        const imageId = stack.imageIds[stack.currentImageIndex];
        const originalImage = await cornerstone.loadAndCacheImage(imageId);
        let displayImage = originalImage;

        if (viewportIndex > 0 && showMasks[viewportIndex]) {
          const mask = maskData.get(stack.currentImageIndex);
          if (mask && mask.length === 128 && mask[0].length === 128) {
            const pixelData = originalImage.getPixelData();
            const modifiedPixelData = new Uint16Array(pixelData.length);
            for (let i = 0; i < pixelData.length; i++) {
              const row = Math.floor(i / 128);
              const col = i % 128;
              modifiedPixelData[i] = mask[row][col] ? pixelData[i] : 32768;
            }
            displayImage = {
              ...originalImage,
              getPixelData: () => modifiedPixelData,
            };
          }
        }

        cornerstone.displayImage(element, displayImage);
        const viewport = cornerstone.getViewport(element);
        if (viewport) {
          viewport.scale = 1.0;
          viewport.translation.x = 0;
          viewport.translation.y = 0;

          if (viewportIndex > 0) {
            viewport.voi.windowWidth = 65535;
            viewport.voi.windowCenter = 32768;
            viewport.colormap =
              colormaps[viewportIndex] !== "grayscale"
                ? colormaps[viewportIndex]
                : undefined;
          } else {
            const imageData = originalImage.getPixelData();
            const minPixelValue = Math.min(...imageData);
            const maxPixelValue = Math.max(...imageData);
            if (minPixelValue !== maxPixelValue) {
              const pixelValues = Array.from(imageData).sort((a, b) => a - b);
              const percentile5Index = Math.floor(pixelValues.length * 0.05);
              const percentile95Index = Math.floor(pixelValues.length * 0.95);
              viewport.voi.windowWidth =
                pixelValues[percentile95Index] - pixelValues[percentile5Index];
              viewport.voi.windowCenter =
                (pixelValues[percentile95Index] +
                  pixelValues[percentile5Index]) /
                2;
            } else {
              let windowWidth = 1800,
                windowCenter = 900;
              const windowWidthData = originalImage.data.string("x00281050");
              const windowCenterData = originalImage.data.string("x00281051");
              if (windowWidthData?.value?.length > 0)
                windowWidth = parseFloat(windowWidthData.value[0]);
              if (windowCenterData?.value?.length > 0)
                windowCenter = parseFloat(windowCenterData.value[0]);
              viewport.voi.windowWidth = windowWidth;
              viewport.voi.windowCenter = windowCenter;
            }
            viewport.colormap = undefined;
          }

          viewport.gamma = 1.0;
          cornerstone.setViewport(element, viewport);
          cornerstone.fitToWindow(element);
        }
      } catch (err) {
        console.error(
          `Error loading image for viewport ${viewportIndex}:`,
          err
        );
        setError(`Failed to load image in viewport ${viewportIndex}`);
      }
    },
    [colormaps, showMasks, maskData]
  );

  const loadImageForViewportRef = useRef(loadImageForViewport);
  useEffect(() => {
    loadImageForViewportRef.current = loadImageForViewport;
  }, [loadImageForViewport]);

  useEffect(() => {
    if (!elementsEnabled) return;
    const stackKeys = ["localized", "strain1", "strain2", "strain3"] as const;
    dicomImageRefs.current.forEach((element, index) => {
      const stack = stacks[stackKeys[index]];
      if (stack.imageIds.length > 0) {
        loadImageForViewportRef.current(element, stack, index);
      }
    });
  }, [elementsEnabled, stacks, showMasks]);

  const toggleMask = useCallback(
    (viewportIndex: number) => {
      if (!masks || masks.length === 0) {
        alert("No masks available for this series.");
        return;
      }
      setShowMasks((prev) => {
        const newState = [...prev];
        newState[viewportIndex] = !newState[viewportIndex];
        return newState;
      });
    },
    [masks]
  );

  const handleColormapChange = useCallback(
    (viewportIndex: number, colormap: string) => {
      setColormaps((prev) => {
        const newColormaps = [...prev];
        newColormaps[viewportIndex] = colormap;
        return newColormaps;
      });
    },
    []
  );

  useEffect(() => {
    if (!elementsEnabled) return;
    const stackKeys = ["localized", "strain1", "strain2", "strain3"] as const;
    dicomImageRefs.current.forEach((element, index) => {
      const stack = stacks[stackKeys[index]];
      if (stack.imageIds.length > 0) {
        loadImageForViewportRef.current(element, stack, index);
      }
    });
  }, [elementsEnabled, stacks, showMasks, colormaps]);

  const togglePlayback = useCallback(
    () => setPlaybackState((prev) => ({ ...prev, isPlaying: !prev.isPlaying })),
    []
  );

  const changeSpeed = useCallback(
    (speed: number) => setPlaybackState((prev) => ({ ...prev, speed })),
    []
  );

  const exportSeriesToZip = useCallback(
    async (index: number) => {
      const series = [localizedSeries, strain1, strain2, strain3][index];
      if (!series) return;

      const zip = new JSZip();
      series.forEach((file: SeriesFile) =>
        zip.file(
          file.filename,
          Uint8Array.from(atob(file.content), (c) => c.charCodeAt(0))
        )
      );
      const content = await zip.generateAsync({ type: "blob" });
      saveAs(
        content,
        `${["Localized", "Strain1", "Strain2", "Strain3"][index]}_series.zip`
      );
    },
    [localizedSeries, strain1, strain2, strain3]
  );

  const handlePrev = useCallback(() => {
    setStacks((prev) => {
      const maxIndex =
        Math.min(
          prev.localized.imageIds.length,
          prev.strain1.imageIds.length,
          prev.strain2.imageIds.length,
          prev.strain3.imageIds.length
        ) - 1;
      const newIndex =
        prev.localized.currentImageIndex > 0
          ? prev.localized.currentImageIndex - 1
          : maxIndex;
      setHasInteractedWithWWWC([false, false, false, false]);
      setStrainRanges([
        { min: 0, max: 1 },
        { min: -0.5, max: 0.5 },
        { min: -0.5, max: 0.5 },
        { min: -0.5, max: 0.5 },
      ]);
      setJustChangedFrame(true);
      return {
        localized: { ...prev.localized, currentImageIndex: newIndex },
        strain1: { ...prev.strain1, currentImageIndex: newIndex },
        strain2: { ...prev.strain2, currentImageIndex: newIndex },
        strain3: { ...prev.strain3, currentImageIndex: newIndex },
      };
    });
  }, []);

  const handleNext = useCallback(() => {
    setStacks((prev) => {
      const maxIndex =
        Math.min(
          prev.localized.imageIds.length,
          prev.strain1.imageIds.length,
          prev.strain2.imageIds.length,
          prev.strain3.imageIds.length
        ) - 1;
      const newIndex =
        prev.localized.currentImageIndex < maxIndex
          ? prev.localized.currentImageIndex + 1
          : 0;
      setHasInteractedWithWWWC([false, false, false, false]);
      setStrainRanges([
        { min: 0, max: 1 },
        { min: -0.5, max: 0.5 },
        { min: -0.5, max: 0.5 },
        { min: -0.5, max: 0.5 },
      ]);
      setJustChangedFrame(true);
      return {
        localized: { ...prev.localized, currentImageIndex: newIndex },
        strain1: { ...prev.strain1, currentImageIndex: newIndex },
        strain2: { ...prev.strain2, currentImageIndex: newIndex },
        strain3: { ...prev.strain3, currentImageIndex: newIndex },
      };
    });
  }, []);

  const handleShowBullseyeModal = () => {
    if (!bullseye1 || !bullseye2 || !bullseye3) {
      alert("No Bull's Eye Plots available.");
      return;
    }
    setIsBullseyeModalOpen(true);
  };

  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (playbackState.isPlaying) {
      intervalRef.current = setInterval(() => {
        setStacks((prev) => {
          const maxIndex =
            Math.min(
              prev.localized.imageIds.length,
              prev.strain1.imageIds.length,
              prev.strain2.imageIds.length,
              prev.strain3.imageIds.length
            ) - 1;
          const nextIndex =
            (prev.localized.currentImageIndex + 1) % (maxIndex + 1);
          return {
            localized: { ...prev.localized, currentImageIndex: nextIndex },
            strain1: { ...prev.strain1, currentImageIndex: nextIndex },
            strain2: { ...prev.strain2, currentImageIndex: nextIndex },
            strain3: { ...prev.strain3, currentImageIndex: nextIndex },
          };
        });
      }, 1000 / playbackState.speed);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [playbackState.isPlaying, playbackState.speed]);

  const generateReport = async () => {
    const currentFrame = stacks.localized.currentImageIndex + 1;
    const plotIndex = currentFrame - 1;

    // Validate content fields
    const contents = [
      { data: bullseye1[plotIndex].content, label: "E1 Bullseye" },
      { data: bullseye2[plotIndex].content, label: "E2 Bullseye" },
      { data: bullseye3[plotIndex].content, label: "E3 Bullseye" },
      { data: bullseye1Ring[plotIndex].content, label: "E1 Ring Bullseye" },
      { data: bullseye2Ring[plotIndex].content, label: "E2 Ring Bullseye" },
      { data: bullseye3Ring[plotIndex].content, label: "E3 Ring Bullseye" },
    ];

    for (const { data, label } of contents) {
      if (!data || typeof data !== "string" || data.trim() === "") {
        alert(
          `Cannot generate report: Invalid or missing content for ${label}.`
        );
        return;
      }
    }

    try {
      const formData = new FormData();

      // Helper to convert base64 to Blob
      const base64ToBlob = (base64, label) => {
        let base64Data = base64;
        let mimeString = "image/png";

        // Handle data URL format or raw base64
        if (base64.startsWith("data:image/png;base64,")) {
          base64Data = base64.split(",")[1];
        } else if (base64.startsWith("data:")) {
          const parts = base64.split(";base64,");
          if (parts.length !== 2) {
            throw new Error(`Invalid data URL format for ${label}`);
          }
          mimeString = parts[0].split(":")[1];
          base64Data = parts[1];
        } // If raw base64, use as-is

        // Validate base64 string
        if (!/^[A-Za-z0-9+/=]+$/.test(base64Data)) {
          throw new Error(`Invalid base64 characters in ${label}`);
        }

        // Ensure base64 length is valid (multiple of 4)
        if (base64Data.length % 4 !== 0) {
          throw new Error(`Invalid base64 length for ${label}`);
        }

        try {
          const byteString = atob(base64Data);
          const ab = new ArrayBuffer(byteString.length);
          const ia = new Uint8Array(ab);
          for (let i = 0; i < byteString.length; i++) {
            ia[i] = byteString.charCodeAt(i);
          }
          return new Blob([ab], { type: mimeString });
        } catch (e) {
          throw new Error(`Failed to decode base64 for ${label}: ${e.message}`);
        }
      };

      // Append Bullseye PNGs for current frame
      formData.append(
        "bullseye1_png",
        base64ToBlob(bullseye1[plotIndex].content, "E1 Bullseye"),
        `bullseye1_frame${currentFrame}.png`
      );
      formData.append(
        "bullseye2_png",
        base64ToBlob(bullseye2[plotIndex].content, "E2 Bullseye"),
        `bullseye2_frame${currentFrame}.png`
      );
      formData.append(
        "bullseye3_png",
        base64ToBlob(bullseye3[plotIndex].content, "E3 Bullseye"),
        `bullseye3_frame${currentFrame}.png`
      );
      formData.append(
        "bullseye1_ring_png",
        base64ToBlob(bullseye1Ring[plotIndex].content, "E1 Ring Bullseye"),
        `bullseye1_ring_frame${currentFrame}.png`
      );
      formData.append(
        "bullseye2_ring_png",
        base64ToBlob(bullseye2Ring[plotIndex].content, "E2 Ring Bullseye"),
        `bullseye2_ring_frame${currentFrame}.png`
      );
      formData.append(
        "bullseye3_ring_png",
        base64ToBlob(bullseye3Ring[plotIndex].content, "E3 Ring Bullseye"),
        `bullseye3_ring_frame${currentFrame}.png`
      );

      // Prepare segment means: [E1_frame2, ..., E1_frame7, E2_frame2, ..., E3_frame7]
      const segmentMeansData = [
        ...segmentMeans.bullseye1.map((frameData) => frameData.segment_means),
        ...segmentMeans.bullseye2.map((frameData) => frameData.segment_means),
        ...segmentMeans.bullseye3.map((frameData) => frameData.segment_means),
      ];
      formData.append("segment_means", JSON.stringify(segmentMeansData));

      // Send request to /report
      const response = await fetch("http://127.0.0.1:8000/report", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        console.log("Report generated successfully for frame:", currentFrame);

        // Handle PDF response correctly
        const blob = await response.blob();
        saveAs(blob, `strain_report_frame_${currentFrame}.pdf`);
      } else {
        const errorText = await response.text();
        console.error("Error generating report:", errorText);
        alert("Failed to generate report. Please try again.");
      }
    } catch (err) {
      console.error("Error generating report:", err);
      alert("Failed to generate report. Please try again.");
    }
  };

  return (
    <div className="flex flex-col bg-gray-900 text-gray-100 min-h-screen p-4 pt-16">
      {error && (
        <div className="bg-red-500 text-white p-4 rounded-md mb-4">{error}</div>
      )}

      {/* Control Bar */}
      <div className="bg-gray-800 p-3 rounded-md mb-4 flex flex-wrap items-center justify-center gap-3">
        <Button
          variant="ghost"
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          onClick={handleShowBullseyeModal}
          disabled={!bullseye1 || !bullseye2 || !bullseye3}
        >
          <Eye className="w-4 h-4" />
          Bull's Eye
        </Button>

        <Button
          variant="ghost"
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          onClick={generateReport}
        >
          <FileText className="w-4 h-4" />
          Generate Report
        </Button>

        <Button
          variant="ghost"
          className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white"
          onClick={togglePlayback}
        >
          {playbackState.isPlaying ? (
            <Pause className="w-4 h-4" />
          ) : (
            <Play className="w-4 h-4" />
          )}
          {playbackState.isPlaying ? "Pause" : "Play"}
        </Button>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            className="bg-gray-700 hover:bg-gray-600 text-white"
            onClick={handlePrev}
          >
            <SkipBack className="w-4 h-4" />
          </Button>

          <Button
            variant="ghost"
            className="bg-gray-700 hover:bg-gray-600 text-white"
            onClick={handleNext}
          >
            <SkipForward className="w-4 h-4" />
          </Button>
        </div>

        <select
          className="bg-gray-700 border border-gray-600 rounded-md px-3 py-1 text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
          value={playbackState.speed}
          onChange={(e) => changeSpeed(Number(e.target.value))}
        >
          <option value={0.5}>0.5x</option>
          <option value={1}>1x</option>
          <option value={2}>2x</option>
          <option value={4}>4x</option>
        </select>
      </div>

      {/* Viewers Grid */}
      <div className="flex justify-center mb-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-9/12">
          {["Localized", "Strain1", "Strain2", "Strain3"].map(
            (title, index) => (
              <div
                key={title}
                className="bg-gray-800 p-3 rounded-md border border-gray-700"
              >
                <div className="flex justify-between items-center mb-2">
                  <h3 className="text-lg font-semibold text-blue-300">
                    {title}
                  </h3>

                  <div className="flex gap-2">
                    {index > 0 && (
                      <>
                        <select
                          className="bg-gray-700 border border-gray-600 rounded-md px-2 py-1 text-xs text-white"
                          value={colormaps[index]}
                          onChange={(e) =>
                            handleColormapChange(index, e.target.value)
                          }
                        >
                          <option value="jet">Jet</option>
                          <option value="hot">Hot</option>
                          <option value="cool">Cool</option>
                        </select>

                        <Button
                          variant="ghost"
                          className={`px-2 py-1 text-xs ${
                            showMasks[index]
                              ? "bg-blue-600 hover:bg-blue-700 text-white"
                              : "bg-gray-700 hover:bg-gray-600 text-white"
                          }`}
                          onClick={() => toggleMask(index)}
                          disabled={!masks || masks.length === 0}
                        >
                          {showMasks[index] ? (
                            <EyeOff className="w-3 h-3" />
                          ) : (
                            <Eye className="w-3 h-3" />
                          )}
                        </Button>
                      </>
                    )}

                    <Button
                      variant="ghost"
                      className="px-2 py-1 bg-green-600 hover:bg-green-700 text-white text-xs"
                      onClick={() => exportSeriesToZip(index)}
                    >
                      <Download className="w-3 h-3" />
                    </Button>
                  </div>
                </div>

                <div className="relative flex">
                  <div
                    ref={(el) => (dicomImageRefs.current[index] = el)}
                    className="w-84 h-64 border border-gray-700 rounded-md bg-black relative"
                  >
                    {stacks[
                      ["localized", "strain1", "strain2", "strain3"][
                        index
                      ] as keyof typeof stacks
                    ].imageIds.length === 0 && (
                      <p className="text-center text-gray-400 mt-20">
                        Loading...
                      </p>
                    )}

                    <p className="absolute bottom-2 left-2 text-xs text-gray-300 bg-gray-900/80 px-2 py-1 rounded">
                      Frame{" "}
                      {stacks[
                        ["localized", "strain1", "strain2", "strain3"][
                          index
                        ] as keyof typeof stacks
                      ].currentImageIndex + 1}{" "}
                      /{" "}
                      {
                        stacks[
                          ["localized", "strain1", "strain2", "strain3"][
                            index
                          ] as keyof typeof stacks
                        ].imageIds.length
                      }
                    </p>
                  </div>

                  {index > 0 &&
                    colormaps[index] !== "grayscale" &&
                    isStrainRangesReady &&
                    strainRanges[index] && (
                      <div className="ml-2 flex items-start">
                        <canvas
                          ref={(el) => (colorBarRefs.current[index] = el)}
                          width={20}
                          className="border border-gray-600 h-full"
                        />
                        <div className="ml-1 flex flex-col justify-between h-full text-xs">
                          <span className="text-gray-300">
                            {strainRanges[index].max.toFixed(4)}
                          </span>
                          <div className="relative group">
                            <Info className="w-3 h-3 text-gray-400" />
                            <div className="absolute hidden group-hover:block left-full ml-1 w-48 bg-gray-800 text-xs text-gray-300 p-2 rounded-md shadow-lg z-10">
                              Gradient represents the full strain range; labels
                              show current WW/WC range.
                            </div>
                          </div>
                          <span className="text-gray-300">
                            {strainRanges[index].min.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    )}
                </div>
              </div>
            )
          )}
        </div>
      </div>

      <FrameBullseyeModal
        open={isBullseyeModalOpen}
        onOpenChange={setIsBullseyeModalOpen}
        currentFrame={stacks.localized.currentImageIndex + 2}
        bullseye1={bullseye1}
        bullseye2={bullseye2}
        bullseye3={bullseye3}
        bullseye1Ring={bullseye1Ring}
        bullseye2Ring={bullseye2Ring}
        bullseye3Ring={bullseye3Ring}
      />

      {isLoading && (
        <div className="fixed inset-0 flex justify-center items-center bg-gray-900/80 z-50">
          <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-blue-500"></div>
        </div>
      )}
    </div>
  );
};

export default ImageViewer;
