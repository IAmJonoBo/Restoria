"use client";
import React from "react";
import CompareSlider from "../components/CompareSlider";
import Queue from "../components/Queue";
import UploadDropzone from "../components/UploadDropzone";

type JobStatus = {
  id: string;
  status: string;
  progress: number;
  result_count: number;
  results_path?: string;
  error?: string;
};

export default function Page() {
  const [inputPath, setInputPath] = React.useState("inputs/whole_imgs");
  const [job, setJob] = React.useState<JobStatus | null>(null);
  const [preset, setPreset] = React.useState("natural");
  const [metrics, setMetrics] = React.useState("off");
  const [background, setBackground] = React.useState("none");
  const [events, setEvents] = React.useState<any[]>([]);
  const [images, setImages] = React.useState<{ input: string; output: string }[]>([]);
  const wsRef = React.useRef<WebSocket | null>(null);

  async function submit() {
    const res = await fetch("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: inputPath, backend: "gfpgan", background, preset, metrics, output: "results", dry_run: true })
    });
    if (!res.ok) return alert("Failed to submit job");
    const js: JobStatus = await res.json();
    setJob(js);

    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/jobs/${js.id}/stream`);
    wsRef.current = ws;
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        setEvents((prev) => [...prev, msg]);
        if (msg.type === "image") {
          setImages((arr) => [...arr, { input: msg.input, output: msg.output }]);
        }
      } catch {}
    };
    ws.onclose = () => {
      wsRef.current = null;
    };
  }

  return (
    <div style={{ display: "grid", gap: 24 }}>
      <h1>GFPP — Face/Scene Restoration</h1>
      <div style={{ display: "grid", gap: 8 }}>
        <label>
          Input path (server):
          <input value={inputPath} onChange={(e) => setInputPath(e.target.value)} style={{ marginLeft: 8 }} />
        </label>
        <label>
          Preset:
          <select value={preset} onChange={(e) => setPreset(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="natural">Natural</option>
            <option value="detail">Detail</option>
            <option value="document">Document</option>
          </select>
        </label>
        <label>
          Metrics:
          <select value={metrics} onChange={(e) => setMetrics(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="off">Off</option>
            <option value="fast">Fast</option>
            <option value="full">Full</option>
          </select>
        </label>
        <label>
          Background:
          <select value={background} onChange={(e) => setBackground(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="none">None</option>
            <option value="realesrgan">RealESRGAN</option>
          </select>
        </label>
        <button onClick={submit}>Submit Dry-Run Job</button>
      </div>
      <Queue />
      {images.length > 0 && (
        <section>
          <h2>Results</h2>
          {images.slice(0, 1).map((r, i) => (
            <CompareSlider
              key={i}
              before={`/file?path=${encodeURIComponent(r.input)}`}
              after={`/file?path=${encodeURIComponent(r.output)}`}
            />
          ))}
        </section>
      )}
      <section>
        <h2>Events</h2>
        <pre style={{ maxHeight: 280, overflow: "auto", background: "#111", color: "#e5e7eb", padding: 8 }}>
          {events.map((e, i) => (
            <div key={i}>{JSON.stringify(e)}</div>
          ))}
        </pre>
      </section>
    </div>
  );
}
