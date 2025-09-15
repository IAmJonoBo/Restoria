import React from "react";

type Props = { metrics?: Record<string, any> };

function fmtVal(v: any): string {
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(3);
  }
  return String(v);
}

export default function MetricsCard({ metrics }: Props) {
  if (!metrics) return null;
  const entries = Object.entries(metrics).filter(([k, v]) => v !== undefined && v !== null);
  if (entries.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 13, color: "#374151" }}>
      {entries.map(([k, v]) => {
        if (k === "identity_retry" && v) {
          return (
            <span key={k} style={{ background: "#dcfce7", color: "#166534", padding: "2px 6px", borderRadius: 4 }}>
              Identity Lock applied
            </span>
          );
        }
        if (typeof v === "boolean") {
          return (
            <span key={k} style={{ background: "#f3f4f6", padding: "2px 6px", borderRadius: 4 }}>
              <b style={{ color: "#111827" }}>{k}</b>: {v ? "✓" : "✗"}
            </span>
          );
        }
        return (
          <span key={k} style={{ background: "#f3f4f6", padding: "2px 6px", borderRadius: 4 }}>
            <b style={{ color: "#111827" }}>{k}</b>: {fmtVal(v)}
          </span>
        );
      })}
    </div>
  );
}
