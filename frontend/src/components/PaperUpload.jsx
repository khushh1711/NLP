import { useRef, useState, useCallback, useEffect } from "react";
import { uploadPaper } from "../services/api";


/* -------------------------------------------------------
   Loading progress steps shown while the backend processes
   ------------------------------------------------------- */
const PROGRESS_STEPS = [
  { id: "upload",    icon: "📤", label: "Uploading PDF to server…" },
  { id: "extract",   icon: "📄", label: "Extracting text with PyMuPDF…" },
  { id: "clean",     icon: "🧹", label: "Cleaning and preprocessing text…" },
  { id: "segment",   icon: "✂️",  label: "Segmenting document regions…" },
  { id: "summarize", icon: "🤖", label: "Running T5 summarization model…" },
  { id: "done",      icon: "✅", label: "Finalizing results…" },
];


/* -------------------------------------------------------
   Summary section definitions
   ------------------------------------------------------- */
const SUMMARY_SECTIONS = [
  {
    key:       "overview",
    modifier:  "overview",
    icon:      "🔭",
    label:     "Overview",
    emptyMsg:  "No overview generated for this document.",
  },
  {
    key:       "method",
    modifier:  "method",
    icon:      "⚙️",
    label:     "Method",
    emptyMsg:  "No method summary generated.",
  },
  {
    key:       "results",
    modifier:  "results",
    icon:      "📊",
    label:     "Results",
    emptyMsg:  "No results summary generated.",
  },
  {
    key:       "conclusion",
    modifier:  "conclusion",
    icon:      "🏁",
    label:     "Conclusion",
    emptyMsg:  "No conclusion summary generated.",
  },
];


/* -------------------------------------------------------
   Helper: format bytes as MB
   ------------------------------------------------------- */
function formatSize(bytes) {
  if (bytes < 1024 * 1024) {
    return (bytes / 1024).toFixed(1) + " KB";
  }
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

/* -------------------------------------------------------
   Helper: format large numbers with comma separators
   ------------------------------------------------------- */
function formatNumber(n) {
  return n?.toLocaleString() ?? "—";
}


/* -------------------------------------------------------
   Animated progress stepper (shown while loading)
   ------------------------------------------------------- */
function ProgressStepper({ activeIndex }) {
  return (
    <div className="progress-steps">
      {PROGRESS_STEPS.map((step, i) => {
        const isDone   = i < activeIndex;
        const isActive = i === activeIndex;
        return (
          <div
            key={step.id}
            className={
              "progress-step" +
              (isDone   ? " step-done"   : "") +
              (isActive ? " step-active" : "")
            }
          >
            <div className="step-icon">
              {isDone ? "✓" : step.icon}
            </div>
            <span className="step-text">{step.label}</span>
          </div>
        );
      })}
    </div>
  );
}


/* -------------------------------------------------------
   Main PaperUpload component
   ------------------------------------------------------- */
function PaperUpload() {

  const [selectedFile, setSelectedFile]   = useState(null);
  const [result,       setResult]         = useState(null);
  const [loading,      setLoading]        = useState(false);
  const [error,        setError]          = useState("");
  const [dragOver,     setDragOver]       = useState(false);
  const [progressIdx,  setProgressIdx]    = useState(0);

  const fileInputRef = useRef(null);
  const progressTimer = useRef(null);


  /* Advance the progress stepper while loading */
  useEffect(() => {
    if (loading) {
      setProgressIdx(0);
      let idx = 0;
      const delays = [400, 1200, 2000, 3200, 4800, 6500];

      delays.forEach((delay, i) => {
        progressTimer.current = setTimeout(() => {
          setProgressIdx(i);
        }, delay);
      });
    } else {
      clearAllTimers();
    }

    return () => clearAllTimers();
  }, [loading]);

  function clearAllTimers() {
    if (progressTimer.current) {
      clearTimeout(progressTimer.current);
    }
  }


  /* File validation and selection */
  const selectFile = useCallback((file) => {
    if (!file) return;

    if (file.type !== "application/pdf") {
      setError("Only PDF files are supported. Please select a .pdf file.");
      setSelectedFile(null);
      return;
    }

    if (file.size > 50 * 1024 * 1024) {
      setError("File size exceeds 50 MB. Please upload a smaller PDF.");
      setSelectedFile(null);
      return;
    }

    setSelectedFile(file);
    setResult(null);
    setError("");
  }, []);


  /* Input change handler */
  const handleFileChange = (e) => {
    selectFile(e.target.files[0]);
  };


  /* Drag and drop handlers */
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    selectFile(file);
  };


  /* Upload and analyze */
  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select a PDF file first.");
      return;
    }

    try {
      setLoading(true);
      setError("");
      setResult(null);

      const data = await uploadPaper(selectedFile);

      console.log("Backend result:", data);
      setResult(data);

    } catch (err) {
      console.error(err);

      const detail = err?.response?.data?.detail;
      setError(
        detail
          ? `Error: ${detail}`
          : "Something went wrong while analyzing the PDF. Please try again."
      );

    } finally {
      setLoading(false);
    }
  };


  /* Reset state */
  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setError("");
    setLoading(false);
    setProgressIdx(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };


  /* ---- Render ---- */
  return (
    <section className="analysis-card" id="analysis-section">

      {/* ---- Header ---- */}
      <div className="upload-header">
        <h2>Analyze a Research Paper</h2>
        <p>
          Upload a scientific PDF to extract structured insights using a
          Transformer-based academic summarization model (T5, fine-tuned on arXiv).
        </p>
      </div>


      {/* ---- Drop Zone ---- */}
      <div
        id="upload-dropzone"
        className={
          "upload-dropzone" +
          (dragOver    ? " drag-over" : "") +
          (selectedFile ? " has-file"  : "")
        }
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* Hidden file input covers the whole zone */}
        <input
          id="pdf-file-input"
          ref={fileInputRef}
          type="file"
          accept=".pdf,application/pdf"
          className="upload-file-input"
          onChange={handleFileChange}
        />

        <div className="upload-icon-wrapper">
          {selectedFile ? "📄" : "☁️"}
        </div>

        <div className="upload-title">
          {selectedFile
            ? selectedFile.name
            : dragOver
              ? "Release to upload"
              : "Drag & drop your PDF here"}
        </div>

        {!selectedFile && (
          <>
            <div className="upload-subtitle">
              Supports .pdf format up to 50 MB
            </div>
            <div className="upload-browse-hint">
              or click anywhere to browse files
            </div>
          </>
        )}

        {selectedFile && (
          <div className="upload-subtitle">
            {formatSize(selectedFile.size)} · PDF document
          </div>
        )}
      </div>


      {/* ---- Selected file confirmation card ---- */}
      {selectedFile && (
        <div className="selected-file-card">
          <span className="selected-file-icon">📑</span>
          <div className="selected-file-info">
            <div className="selected-file-name">{selectedFile.name}</div>
            <div className="selected-file-size">
              {formatSize(selectedFile.size)} · PDF
            </div>
          </div>
          <span className="selected-file-check">✓</span>
        </div>
      )}


      {/* ---- Action buttons ---- */}
      <div className="action-row">
        <button
          id="analyze-button"
          className="primary-button"
          onClick={handleUpload}
          disabled={!selectedFile || loading}
        >
          {loading ? (
            <>
              <span className="loading-spinner" style={{
                width: "16px",
                height: "16px",
                border: "2px solid rgba(255,255,255,0.3)",
                borderTopColor: "#fff",
                display: "inline-block",
                borderRadius: "50%",
                animation: "spin 0.7s linear infinite"
              }} />
              Analyzing…
            </>
          ) : (
            <>🔬 Analyze Paper</>
          )}
        </button>

        {(result || error) && (
          <button
            id="reset-button"
            className="secondary-button"
            onClick={handleReset}
          >
            ← Analyze Another Paper
          </button>
        )}
      </div>


      {/* ---- Error ---- */}
      {error && (
        <div className="error-message" id="error-display" role="alert">
          <span>⚠️</span>
          <span>{error}</span>
        </div>
      )}


      {/* ---- Loading progress ---- */}
      {loading && (
        <div className="loading-container" id="loading-container">
          <div className="loading-spinner-wrapper">
            <div className="loading-spinner" />
          </div>
          <div className="loading-title">Processing your paper…</div>
          <div className="loading-subtitle">
            This may take 30–90 seconds depending on PDF length and server load.
          </div>
          <ProgressStepper activeIndex={progressIdx} />
        </div>
      )}


      {/* ---- Results ---- */}
      {result && (
        <>
          {/* Document stats */}
          <div className="document-analysis" id="document-analysis">
            <div className="results-header">
              <div className="results-header-icon">📋</div>
              <h3 className="section-title">Document Analysis</h3>
            </div>

            <div className="document-grid">
              <div className="info-card">
                <span className="info-label">Filename</span>
                <span className="info-value" title={result.filename}>
                  {result.filename}
                </span>
              </div>

              <div className="info-card">
                <span className="info-label">Content Type</span>
                <span className="info-value">{result.content_type}</span>
              </div>

              <div className="info-card">
                <span className="info-label">Extracted Characters</span>
                <span className="info-value accent">
                  {formatNumber(result.text_length)}
                </span>
              </div>

              <div className="info-card">
                <span className="info-label">Extracted Words</span>
                <span className="info-value accent">
                  {formatNumber(result.word_count)}
                </span>
              </div>
            </div>
          </div>


          <div className="results-divider" />


          {/* Summary cards */}
          <div className="summary-container" id="summary-container">
            <div className="summary-title-row">
              <h3 className="summary-title">AI-Generated Summary</h3>
              <span className="ai-badge">⚡ T5 · arXiv fine-tuned</span>
            </div>

            <div className="summary-grid">
              {SUMMARY_SECTIONS.map(({ key, modifier, icon, label, emptyMsg }) => (
                <div
                  key={key}
                  id={`summary-${key}`}
                  className={`summary-section summary-section--${modifier}`}
                >
                  <div className="summary-section-header">
                    <div className="summary-section-icon">{icon}</div>
                    <h4>{label}</h4>
                  </div>
                  <p>
                    {result.summary?.[key]
                      ? result.summary[key]
                      : <em style={{ opacity: 0.5 }}>{emptyMsg}</em>
                    }
                  </p>
                </div>
              ))}
            </div>


            {/* Disclaimer */}
            <div className="disclaimer-banner" id="disclaimer-banner">
              <span className="disclaimer-icon">⚠️</span>
              <p className="disclaimer-text">
                <strong>Important:</strong> This summary is AI-generated using approximate
                document-position segmentation, not reliable section-heading detection.
                The output may contain omissions, inaccuracies, or repeated phrases.
                Always verify claims against the original paper. This tool assists initial
                review — it does not replace careful reading of the source document.
              </p>
            </div>
          </div>
        </>
      )}

    </section>
  );
}


export default PaperUpload;