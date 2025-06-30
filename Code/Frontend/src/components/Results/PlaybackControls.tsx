import React from "react";
import Button from "./Button";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";

interface PlaybackControlsProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  onPrevious: () => void;
  onNext: () => void;
  onSpeedChange: (speed: number) => void;
  currentSpeed: number;
  disabled: boolean;
}

const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  isPlaying,
  onPlayPause,
  onPrevious,
  onNext,
  onSpeedChange,
  currentSpeed,
  disabled,
}) => {
  const speedOptions = [0.5, 1, 1.5, 2];

  return (
    <div className="flex flex-wrap items-center justify-center gap-3 p-4 bg-gray-100 rounded-lg">
      <div className="flex items-center gap-2">
        <Button onClick={onPrevious} disabled={disabled} variant="secondary">
          <SkipBack size={16} />
        </Button>

        <Button onClick={onPlayPause} disabled={disabled}>
          {isPlaying ? <Pause size={16} /> : <Play size={16} />}
        </Button>

        <Button onClick={onNext} disabled={disabled} variant="secondary">
          <SkipForward size={16} />
        </Button>
      </div>

      <div className="flex items-center gap-2 ml-2">
        <span className="text-sm text-blue-700 font-medium">Speed:</span>
        <div className="flex bg-gray-200 rounded-md overflow-hidden">
          {speedOptions.map((speed) => (
            <button
              key={speed}
              onClick={() => onSpeedChange(speed)}
              className={`px-2 py-1 text-sm transition-colors ${
                speed === currentSpeed
                  ? "bg-blue-500 text-white"
                  : "hover:bg-gray-300"
              }`}
              disabled={disabled}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default PlaybackControls;
