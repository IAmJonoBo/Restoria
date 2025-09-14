"use client";
import React from "react";
import CompareSlider from "../components/CompareSlider";
import Queue from "../components/Queue";
import UploadDropzone from "../components/UploadDropzone";
import MetricsCard from "../components/MetricsCard";

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
  const [quality, setQuality] = React.useState("balanced");
  const [autoBackend, setAutoBackend] = React.useState(false);
  const [identityLock, setIdentityLock] = React.useState(false);
  const [backend, setBackend] = React.useState("gfpgan");
  const [modelPathOnnx, setModelPathOnnx] = React.useState("");
  const [events, setEvents] = React.useState<any[]>([]);
  const [images, setImages] = React.useState<{ input: string; output: string; metrics?: Record<string, any> }[]>([]);
  const [done, setDone] = React.useState(false);
  const [manifestPath, setManifestPath] = React.useState<string | null>(null);
  const wsRef = React.useRef<WebSocket | null>(null);

  async function submit() {
    const res = await fetch("/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ input: inputPath, backend, background, preset, quality, metrics, output: "results", dry_run: true, auto_backend: autoBackend, identity_lock: identityLock, model_path_onnx: backend === "gfpgan-ort" ? modelPathOnnx : undefined })
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
          setImages((arr) => [...arr, { input: msg.input, output: msg.output, metrics: msg.metrics }]);
        }
        if (msg.type === "status" && msg.status === "done") {
          setDone(true);
        }
        if (msg.type === "manifest" && msg.path) {
          setManifestPath(String(msg.path));
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
          Backend:
          <select value={backend} onChange={(e) => setBackend(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="gfpgan">GFPGAN (Torch)</option>
            <option value="gfpgan-ort">GFPGAN (ONNX Runtime)</option>
            <option value="codeformer">CodeFormer</option>
            <option value="restoreformerpp">RestoreFormer++</option>
          </select>
        </label>
        {backend === "gfpgan-ort" && (
          <label>
            ONNX model path:
            <input value={modelPathOnnx} onChange={(e) => setModelPathOnnx(e.target.value)} style={{ marginLeft: 8, width: 300 }} placeholder="/path/to/gfpgan.onnx" />
          </label>
        )}
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
        <label>
          Quality:
          <select value={quality} onChange={(e) => setQuality(e.target.value)} style={{ marginLeft: 8 }}>
            <option value="quick">Quick</option>
            <option value="balanced">Balanced</option>
            <option value="best">Best</option>
          </select>
        </label>
        <label>
          <input type="checkbox" checked={autoBackend} onChange={(e) => setAutoBackend(e.target.checked)} /> Auto Backend
        </label>
        <label>
          <input type="checkbox" checked={identityLock} onChange={(e) => setIdentityLock(e.target.checked)} /> Identity Lock
        </label>
        <button onClick={submit}>Submit Dry-Run Job</button>
      </div>
      <Queue />
      {images.length > 0 && (
        <section>
          <h2>Results</h2>
          <div style={{ display: "grid", gap: 16 }}>
            {images.map((r, i) => (
              <div key={i} style={{ display: "grid", gap: 8 }}>
                <CompareSlider
                  before={`/file?path=${encodeURIComponent(r.input)}`}
                  after={`/file?path=${encodeURIComponent(r.output)}`}
                />
                <MetricsCard metrics={r.metrics} />
                <div>
                  <button
                    onClick={async () => {
                      // Re-run only this image using current controls (dry)
                      await fetch(`/jobs/${job?.id}/rerun`, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ input: r.input, backend: "gfpgan", background, preset, quality, metrics, auto_backend: autoBackend, identity_lock: identityLock, dry_run: true }),
                      });
                    }}
                  >
                    Re-run this (dry)
                  </button>
                </div>
              </div>
            ))}
          </div>
          {done && (
            <div style={{ marginTop: 8 }}>
              <a href={`/results/${job?.id}`} target="_blank" rel="noreferrer" style={{ marginRight: 12 }}>
                Download ZIP
              </a>
              {manifestPath && (
                <a href={`/file?path=${encodeURIComponent(manifestPath)}`} target="_blank" rel="noreferrer">
                  View manifest.json
                </a>
              )}
            </div>
          )}
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
