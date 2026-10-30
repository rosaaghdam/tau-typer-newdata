"""Heatmaps as interactive web elements: the same colours as the PNG/PDF figures (figures_apply), plus the exact
value of the person and column under the mouse. Drawn by the browser on a <canvas>; no extra package is needed."""
import json

import numpy as np

from src.figures import nice_region
from src.figures_apply import HEAT_LABEL, heatmap_colors, heatmap_order, heatmap_scale


def matrix_heatmap_html(table, limits, colour_label, row_label, element_id, hover=True, height=560, label_size=11):
    """table: rows x columns, already in display order (index = row names). limits: the colour scale (low, high);
    values outside get the lowest / darkest colour. hover=False shows no values (e.g. for the training cohort,
    whose individual values stay private)."""
    values = table.to_numpy(float)
    data = {"ids": [str(i) for i in table.index], "cols": [str(c) for c in table.columns],
            "values": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in values] if hover else [],
            "colors": heatmap_colors(values, limits).tolist(), "hover": hover}
    cmap, norm = heatmap_scale(limits)
    gradient = ", ".join("#%02x%02x%02x" % tuple(int(round(255 * v)) for v in cmap(p)[:3])
                         for p in np.linspace(1, 0, 21))                       # top of the bar = high
    tick_values = ([t for t in (limits[0], 0, 2, 5, limits[1]) if limits[0] <= t <= limits[1]]
                   if limits[0] < 0 < limits[1] else np.linspace(*limits, 6))
    ticks = "".join(f'<span style="position:absolute; top:{100 * (1 - float(norm(t))):.1f}%; left:22px; '
                    f'transform:translateY(-50%);">{t:.3g}</span>' for t in tick_values)
    names = "".join(f'<div style="writing-mode:vertical-rl; transform:rotate(180deg); text-align:right; '
                    f'margin:4px auto 0; white-space:nowrap;">{name}</div>' for name in data["cols"])
    return f"""
<div id="{element_id}" style="position:relative; display:flex; gap:14px; align-items:stretch;">
  <div style="writing-mode:vertical-rl; transform:rotate(180deg); font-size:12px; color:#52606D; text-align:center;">
    {row_label}</div>
  <div style="flex:1; min-width:0;">
    <canvas style="width:100%; height:{height}px; display:block; image-rendering:pixelated;
      cursor:{'crosshair' if hover else 'default'};"></canvas>
    <div style="display:grid; grid-template-columns:repeat({len(data['cols'])}, minmax(0, 1fr)); font-size:{label_size}px;
      line-height:1; color:#52606D;">{names}</div>
  </div>
  <div style="display:flex; gap:6px; height:{height}px;">
    <div style="position:relative; width:16px; background:linear-gradient({gradient}); font-size:11px; color:#52606D;">
      {ticks}</div>
    <div style="writing-mode:vertical-rl; transform:rotate(180deg); font-size:12px; color:#1F2933; margin-left:30px;
      text-align:center;">{colour_label}</div>
  </div>
  <div class="tt-heat-tip" style="position:absolute; display:none; pointer-events:none; background:#1F2933; color:#fff;
    font-size:12px; padding:4px 8px; border-radius:4px; white-space:nowrap; z-index:10;"></div>
</div>
<script>
(function() {{
  const box = document.getElementById("{element_id}"), d = {json.dumps(data)};
  const canvas = box.querySelector("canvas"), tip = box.querySelector(".tt-heat-tip");
  const nRows = d.ids.length, nCols = d.cols.length;
  canvas.width = nCols; canvas.height = nRows;                   // one canvas pixel per row and column
  const ctx = canvas.getContext("2d");
  for (let r = 0; r < nRows; r++) for (let c = 0; c < nCols; c++) {{
    ctx.fillStyle = d.colors[r][c]; ctx.fillRect(c, r, 1, 1);
  }}
  if (!d.hover) return;
  canvas.addEventListener("mousemove", function(e) {{
    const rect = canvas.getBoundingClientRect(), outer = box.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / rect.width * nCols);
    const r = Math.floor((e.clientY - rect.top) / rect.height * nRows);
    if (r < 0 || r >= nRows || c < 0 || c >= nCols) {{ tip.style.display = "none"; return; }}
    const v = d.values[r][c];
    tip.textContent = d.ids[r] + " · " + d.cols[c] + ": " + (v === null ? "missing" : v.toFixed(3));
    tip.style.left = (e.clientX - outer.left + 12) + "px"; tip.style.top = (e.clientY - outer.top + 12) + "px";
    tip.style.display = "block";
  }});
  canvas.addEventListener("mouseleave", function() {{ tip.style.display = "none"; }});
}})();
</script>"""


def heatmap_html(z, lobes, limits, element_id="tt_heat"):
    """Step 2: the lobar scores, people sorted by their mean score (lowest at the top)."""
    rows = heatmap_order(z, lobes)
    table = z.loc[rows, lobes].copy()
    table.columns = [nice_region(l) for l in lobes]
    return matrix_heatmap_html(table, limits, f"{HEAT_LABEL} (white up to 0, blue from 2, black at {limits[1]:g})", f"people (n = {len(rows)}), sorted by mean score "
                               "(lowest at the top)", element_id)
