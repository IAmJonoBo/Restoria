import React from "react";

type Props = { metrics?: Record<string, any> };

export default function MetricsCard({ metrics }: Props) {
  if (!metrics) return null;
  const entries = Object.entries(metrics).filter(([k, v]) => v !== undefined && v !== null);
  if (entries.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 12, flexWrap: "wrap", fontSize: 13, color: "#374151" }}>
      {entries.map(([k, v]) => (
        <span key={k} style={{ background: "#f3f4f6", padding: "2px 6px", borderRadius: 4 }}>
          <b style={{ color: "#111827" }}>{k}</b>: {String(v)}
        </span>
      ))}
    </div>
  );
}

