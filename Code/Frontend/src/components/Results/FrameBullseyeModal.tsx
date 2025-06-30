import React, { useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Button } from "../ui/button";
import { Download, RefreshCw } from "lucide-react";

interface BullseyeImage {
  filename: string;
  content: string;
}

interface CurrentFrameBullseyeModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  currentFrame: number;
  bullseye1: BullseyeImage[] | null;
  bullseye2: BullseyeImage[] | null;
  bullseye3: BullseyeImage[] | null;
  bullseye1Ring?: BullseyeImage[] | null;
  bullseye2Ring?: BullseyeImage[] | null;
  bullseye3Ring?: BullseyeImage[] | null;
}

const FrameBullseyeModal: React.FC<CurrentFrameBullseyeModalProps> = ({
  open,
  onOpenChange,
  currentFrame,
  bullseye1,
  bullseye2,
  bullseye3,
  bullseye1Ring,
  bullseye2Ring,
  bullseye3Ring,
}) => {
  const [showRings, setShowRings] = useState(false);

  const downloadImage = (imageData: string, filename: string) => {
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${imageData}`;
    link.download = filename;
    link.click();
  };

  // Check if ring plots are available
  const hasRingPlots =
    (bullseye1Ring?.length ?? 0) > 0 &&
    (bullseye2Ring?.length ?? 0) > 0 &&
    (bullseye3Ring?.length ?? 0) > 0;

  const hasPlots =
    currentFrame > 1 &&
    (showRings
      ? hasRingPlots
      : (bullseye1?.length ?? 0) > 0 ||
        (bullseye2?.length ?? 0) > 0 ||
        (bullseye3?.length ?? 0) > 0);
  const plotIndex = currentFrame - 2;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl p-6 bg-gray-800 border border-gray-700 rounded-lg shadow-xl">
        <DialogHeader className="flex justify-between items-center">
          <div className="flex items-center gap-4">
            <DialogTitle className="text-xl font-semibold text-blue-300">
              Bull's Eye Plots - Frame {currentFrame - 1}
            </DialogTitle>

            {hasRingPlots && (
              <Button
                variant="ghost"
                className="flex items-center gap-2 bg-gray-700 hover:bg-gray-600 text-white"
                onClick={() => setShowRings(!showRings)}
              >
                <RefreshCw className="w-4 h-4" />
                {showRings ? "Show Standard" : "Show Rings"}
              </Button>
            )}
          </div>
        </DialogHeader>

        <div className="mt-4">
          {!hasPlots ? (
            <div className="text-center text-gray-400 p-4 bg-gray-700 rounded-md">
              No Bull's Eye Plots available for this frame.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {[
                {
                  key: "E1",
                  data: showRings ? bullseye1Ring : bullseye1,
                  label: `Strain E1 ${showRings ? "(Ring)" : ""}`,
                },
                {
                  key: "E2",
                  data: showRings ? bullseye2Ring : bullseye2,
                  label: `Strain E2 ${showRings ? "(Ring)" : ""}`,
                },
                {
                  key: "E3",
                  data: showRings ? bullseye3Ring : bullseye3,
                  label: `Strain E3 ${showRings ? "(Ring)" : ""}`,
                },
              ].map(({ key, data, label }) => (
                <div key={key} className="flex flex-col items-center">
                  <h3 className="text-md font-medium text-gray-300 mb-3">
                    {label}
                  </h3>
                  {data && data[plotIndex] ? (
                    <div className="relative group">
                      <img
                        src={`data:image/png;base64,${data[plotIndex].content}`}
                        alt={data[plotIndex].filename}
                        className="max-w-full h-auto rounded-md border border-gray-600 shadow-lg transition-transform group-hover:scale-105"
                        style={{ maxWidth: "250px" }}
                      />
                      <Button
                        variant="ghost"
                        className="absolute bottom-3 right-3 bg-gray-700/90 hover:bg-gray-600/90 text-white p-2 rounded-full shadow-md"
                        onClick={() =>
                          downloadImage(
                            data[plotIndex].content,
                            data[plotIndex].filename
                          )
                        }
                      >
                        <Download className="w-4 h-4" />
                      </Button>
                    </div>
                  ) : (
                    <div className="text-red-400 bg-gray-700/50 p-2 rounded-md">
                      Plot not available
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};
export default FrameBullseyeModal;
