from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _esc(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def write_html_report(
    output_dir: str, results: List[Dict[str, Any]], metrics_keys: Optional[List[str]] = None, path: Optional[str] = None
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if metrics_keys is None:
        metrics_keys = sorted({k for r in results for k in (r.get("metrics") or {}).keys()})

    rows = []
    for r in results:
        inp = r.get("input")
        out = r.get("restored_img")
        m = r.get("metrics", {})
        cells = "".join(f"<td>{_esc(m.get(k, ''))}</td>" for k in metrics_keys)
        rows.append(f"<tr>" f"<td>{_esc(inp)}</td>" f"<td>{_esc(out)}</td>" f"{cells}" f"</tr>")

    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset='utf-8'/>
    <title>GFPP Report</title>
    <style>
      body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px; text-align: left; }}
      th {{ background: #f9fafb; }}
    </style>
  </head>
  <body>
    <h1>GFPP Report</h1>
    <table>
      <thead>
        <tr>
          <th>Input</th>
          <th>Restored</th>
          {''.join(f'<th>{_esc(k)}</th>' for k in metrics_keys)}
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </body>
</html>
"""
    out_path = path or os.path.join(output_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


__all__ = ["write_html_report"]
