// Shared utility: format a UTC ISO string as local date
export function formatDate(iso) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      year: "numeric", month: "short", day: "numeric",
    });
  } catch { return iso; }
}

// Format bytes to KB / MB
export function formatSize(bytes) {
  if (!bytes) return "—";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1024 / 1024).toFixed(2) + " MB";
}

// Capitalize first letter
export function capitalize(str) {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// Analysis status -> display label + color class
export const STATUS_META = {
  pending:    { label: "Pending",    cls: "status-pending"    },
  processing: { label: "Processing", cls: "status-processing" },
  completed:  { label: "Analyzed",   cls: "status-done"       },
  failed:     { label: "Failed",     cls: "status-error"      },
};

// Screening decision -> label + color
export const DECISION_META = {
  unreviewed:  { label: "Unreviewed",  cls: "dec-unreviewed"  },
  relevant:    { label: "Relevant",    cls: "dec-relevant"    },
  irrelevant:  { label: "Irrelevant",  cls: "dec-irrelevant"  },
  needs_review:{ label: "Needs Review",cls: "dec-needs-review"},
};

// Field type -> icon
export const FIELD_ICONS = {
  TASK:       "🎯",
  METHOD:     "⚙️",
  DATASET:    "📦",
  METRIC:     "📏",
  RESULT:     "📊",
  LIMITATION: "⚠️",
};

// Verification status -> icon
export const VERIFY_ICONS = {
  needs_review: "🔍",
  verified:     "✅",
  incorrect:    "❌",
};
