"use client";
import React from "react";

type Job = {
  id: string;
  status: string;
  progress: number;
  result_count: number;
  results_path?: string;
  error?: string;
};

export default function Queue() {
  const [jobs, setJobs] = React.useState<Job[]>([]);
  const [tick, setTick] = React.useState(0);

  React.useEffect(() => {
    let active = true;
    async function load() {
      try {
        const r = await fetch("/jobs");
        if (!r.ok) return;
        const js = await r.json();
        if (active) setJobs(js);
      } catch {}
    }
    load();
    const t = setInterval(() => {
      setTick((x) => x + 1);
      load();
    }, 1000);
    return () => {
      active = false;
      clearInterval(t);
    };
  }, []);

  if (jobs.length === 0) return null;

  return (
    <section>
      <h2>Queue</h2>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={{ textAlign: "left", padding: 8 }}>ID</th>
            <th style={{ textAlign: "left", padding: 8 }}>Status</th>
            <th style={{ textAlign: "left", padding: 8 }}>Progress</th>
            <th style={{ textAlign: "left", padding: 8 }}>Results</th>
            <th style={{ textAlign: "left", padding: 8 }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((j) => (
            <tr key={j.id} style={{ borderBottom: "1px solid #e5e7eb" }}>
              <td style={{ padding: 8 }}>{j.id}</td>
              <td style={{ padding: 8 }}>{j.status}</td>
              <td style={{ padding: 8 }}>
                <div style={{ width: 160, background: "#e5e7eb", height: 8, position: "relative" }}>
                  <div
                    style={{
                      width: `${Math.round((j.progress || 0) * 100)}%`,
                      height: 8,
                      background: "#60a5fa",
                      transition: "width 180ms ease-out",
                    }}
                  />
                </div>
                <small>{Math.round((j.progress || 0) * 100)}%</small>
              </td>
              <td style={{ padding: 8 }}>{j.result_count}</td>
              <td style={{ padding: 8, display: "flex", gap: 8 }}>
                {j.status === "done" && (
                  <a href={`/results/${j.id}`} target="_blank" rel="noreferrer">
                    Download ZIP
                  </a>
                )}
                <button
                  onClick={async () => {
                    await fetch(`/jobs/${j.id}/rerun`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dry_run: true }) });
                  }}
                >
                  Re-run (dry)
                </button>
                {j.error && <span style={{ color: "#ef4444" }}>Error: {j.error}</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
