const DicomMetadata = ({ metadata, position }: any) => {
  if (!metadata) return null;

  const overlayStyle =
    "absolute bg-gray-900/90 p-2 rounded text-xs font-mono border border-gray-700 shadow-lg";
  const textStyle = "text-green-400 leading-tight";
  const labelStyle = "text-gray-400 inline-block w-8";

  return (
    <>
      {/* Top-left corner - positioned relative to viewer */}
      <div className={`${overlayStyle} top-2 left-2 max-w-xs`}>
        <p className={textStyle}>
          <span className={labelStyle}>SN:</span>{" "}
          {metadata.SeriesNumber || "N/A"}
        </p>
        <p className={textStyle}>
          <span className={labelStyle}>IN:</span>{" "}
          {metadata.InstanceNumber || "N/A"}
        </p>
      </div>

      {/* Top-right corner - positioned relative to viewer */}
      <div className={`${overlayStyle} top-2 right-2 text-right max-w-xs`}>
        <p className={textStyle}>
          <span className={labelStyle}>PT:</span>{" "}
          {metadata.PatientName || "N/A"}
        </p>
        <p className={textStyle}>
          <span className={labelStyle}>ID:</span> {metadata.PatientID || "N/A"}
        </p>
      </div>

      {/* Bottom-left corner - positioned relative to viewer */}
      <div className={`${overlayStyle} bottom-2 left-2 max-w-xs`}>
        <p className={textStyle}>
          <span className={labelStyle}>POS:</span> ({position?.x || 0},{" "}
          {position?.y || 0})
        </p>
        <p className={textStyle}>
          <span className={labelStyle}>WL:</span>{" "}
          {metadata.WindowLevel || "N/A"}
        </p>
        <p className={textStyle}>
          <span className={labelStyle}>WW:</span>{" "}
          {metadata.WindowWidth || "N/A"}
        </p>
      </div>

      {/* Bottom-right corner - positioned relative to viewer */}
      <div className={`${overlayStyle} bottom-2 right-2 text-right max-w-xs`}>
        <p className={textStyle}>
          <span className={labelStyle}>MOD:</span> {metadata.Modality || "N/A"}
        </p>
        <p className={textStyle}>
          <span className={labelStyle}>DT:</span>{" "}
          {metadata.ContentDate || "N/A"}
        </p>
      </div>
    </>
  );
};
export default DicomMetadata;
