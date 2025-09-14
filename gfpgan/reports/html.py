from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


def write_html_report(output_dir: str, manifest: Dict[str, Any], report_path: Optional[str] = None) -> str:
    """Write a simple comparison report and return the path.

    Shows each input with its first restored image and any available metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: List[Dict[str, Any]] = manifest.get("results", [])
    metrics_file: Optional[str] = manifest.get("metrics_file")
    metrics_lookup: Dict[str, Dict[str, Any]] = {}
    if metrics_file and os.path.isfile(metrics_file):
        import json

        try:
            with open(metrics_file, "r") as f:
                md = json.load(f)
            # Build quick lookup by input
            for rec in md.get("metrics", []):
                metrics_lookup[rec.get("input")] = rec
        except Exception:
            pass

    rows: List[str] = []
    for rec in results:
        inp = rec.get("input")
        outs = rec.get("restored_imgs") or []
        out0 = outs[0] if outs and outs[0] else None
        m = metrics_lookup.get(inp, {})
        idc = m.get("identity_cosine")
        lp = m.get("lpips_alex")
        rows.append(
            f"<tr>\n"
            f"  <td><div class='imgcell'><div>Input</div><img src='{_escape(os.path.relpath(inp, output_dir))}'/></div></td>\n"
            f"  <td><div class='imgcell'><div>Restored</div>{('<img src=\'%s\'/>' % _escape(os.path.relpath(out0, output_dir))) if out0 else '<em>n/a</em>'}</div></td>\n"
            f"  <td><div class='metrics'>"
            f"    <div><b>ArcFace</b>: {idc if idc is not None else 'n/a'}</div>"
            f"    <div><b>LPIPS</b>: {lp if lp is not None else 'n/a'}</div>"
            f"  </div></td>\n"
            f"</tr>"
        )

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset='utf-8'/>
  <title>GFPGAN Report</title>
  <style>
    body {{ font-family: -apple-system, system-ui, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 12px; vertical-align: top; }}
    th {{ text-align: left; background: #f9fafb; font-weight: 600; }}
    .imgcell img {{ max-width: 360px; height: auto; display: block; border: 1px solid #e5e7eb; }}
    .metrics div {{ margin-bottom: 4px; }}
    .meta {{ color: #6b7280; margin-bottom: 16px; }}
  </style>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <meta name='color-scheme' content='light dark'/>
  <style media='(prefers-color-scheme: dark)'>
    body {{ background: #0b0f14; color: #d1d5db; }}
    th, td {{ border-bottom-color: #1f2937; }}
    th {{ background: #0f172a; }}
    .imgcell img {{ border-color: #1f2937; }}
  </style>
  <meta name='robots' content='noindex'/>
  <meta name='referrer' content='no-referrer'/>
  <meta name='viewport' content='width=device-width, initial-scale=1'/>
  <meta http-equiv='X-Content-Type-Options' content='nosniff'/>
  <meta http-equiv='Referrer-Policy' content='no-referrer'/>
  <meta http-equiv='Permissions-Policy' content='interest-cohort=()'/>
  <meta http-equiv='Cross-Origin-Resource-Policy' content='same-origin'/>
  <meta http-equiv='Content-Security-Policy' content="default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline';"/>
  <meta name='generator' content='gfpgan-report-0.1'/>
  <meta name='description' content='GFPGAN results report'/>
  <meta name='theme-color' content='#111827'/>
  <meta charset='utf-8'/>
  <meta http-equiv='X-UA-Compatible' content='IE=edge'/>
  <meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'/>
  <meta name='apple-mobile-web-app-capable' content='yes'/>
  <meta name='apple-mobile-web-app-status-bar-style' content='black-translucent'/>
  <meta name='format-detection' content='telephone=no'/>
  <meta name='apple-mobile-web-app-title' content='GFPGAN Report'/>
  <meta name='application-name' content='GFPGAN Report'/>
  <meta name='msapplication-TileColor' content='#111827'/>
  <meta name='msapplication-config' content='none'/>
  <meta http-equiv='X-UA-Compatible' content='IE=edge'/>
"""
    html += f"""
</head>
<body>
  <h1>GFPGAN Report</h1>
  <div class='meta'>Model: {manifest.get('meta', {}).get('model_name')} • Device: {manifest.get('meta', {}).get('device')}</div>
  <table>
    <thead><tr><th>Input</th><th>Restored</th><th>Metrics</th></tr></thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""

    out_path = report_path or os.path.join(output_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    return out_path


__all__ = ["write_html_report"]

