"""The step 2 heatmap as an interactive web element: the same colours as the PNG/PDF figure
(figures_apply.plot_z_heatmap), plus the exact score of the person and lobe under the mouse.
Drawn by the browser on a <canvas>; no extra package is needed."""
import json

import numpy as np

from src.figures import nice_region
from src.figures_apply import HEAT_LABEL, heatmap_colors, heatmap_order


def heatmap_html(z, lobes, limits, element_id="tt_heat"):
    """HTML + a small script. z: people x lobes (index = person ID). limits: the fixed colour scale (low, high)."""
    rows = heatmap_order(z, lobes)
    values = z.loc[rows, lobes].to_numpy(float)
    data = {"ids": [str(i) for i in rows], "lobes": [nice_region(l) for l in lobes],
            "values": [[None if np.isnan(v) else round(float(v), 3) for v in row] for row in values],
            "colors": heatmap_colors(values, limits).tolist()}
    steps = np.linspace(limits[0], limits[1], 11)
    gradient = ", ".join(heatmap_colors(steps[::-1], limits).tolist())        # top of the bar = high
    ticks = "".join(f'<span style="position:absolute; top:{100 * (limits[1] - t) / (limits[1] - limits[0]):.1f}%; '
                    f'left:22px; transform:translateY(-50%);">{t:g}</span>' for t in np.linspace(*limits, 6))
    return f"""
<div id="{element_id}" style="position:relative; display:flex; gap:14px; align-items:stretch; max-width:760px;">
  <div style="writing-mode:vertical-rl; transform:rotate(180deg); font-size:12px; color:#52606D; text-align:center;">
    people (n = {len(rows)}), sorted by mean score (lowest at the top)</div>
  <div style="flex:1; min-width:0;">
    <canvas style="width:100%; height:560px; display:block; image-rendering:pixelated; cursor:crosshair;"></canvas>
    <div style="display:grid; grid-template-columns:repeat({len(lobes)}, 1fr); font-size:11px; color:#52606D;">
      {"".join(f'<div style="writing-mode:vertical-rl; transform:rotate(180deg); text-align:right; margin:4px auto 0;">{name}</div>' for name in data["lobes"])}
    </div>
  </div>
  <div style="display:flex; gap:6px; height:560px;">
    <div style="position:relative; width:16px; background:linear-gradient({gradient}); font-size:11px; color:#52606D;">
      {ticks}</div>
    <div style="writing-mode:vertical-rl; transform:rotate(180deg); font-size:12px; color:#1F2933; margin-left:22px;
      text-align:center;">{HEAT_LABEL} (below {limits[0]:g} = lowest colour, above {limits[1]:g} = darkest)</div>
  </div>
  <div class="tt-heat-tip" style="position:absolute; display:none; pointer-events:none; background:#1F2933; color:#fff;
    font-size:12px; padding:4px 8px; border-radius:4px; white-space:nowrap; z-index:10;"></div>
</div>
<script>
(function() {{
  const box = document.getElementById("{element_id}"), d = {json.dumps(data)};
  const canvas = box.querySelector("canvas"), tip = box.querySelector(".tt-heat-tip");
  const nRows = d.ids.length, nCols = d.lobes.length;
  canvas.width = nCols; canvas.height = nRows;                   // one canvas pixel per person and lobe
  const ctx = canvas.getContext("2d");
  for (let r = 0; r < nRows; r++) for (let c = 0; c < nCols; c++) {{
    ctx.fillStyle = d.colors[r][c]; ctx.fillRect(c, r, 1, 1);
  }}
  canvas.addEventListener("mousemove", function(e) {{
    const rect = canvas.getBoundingClientRect(), outer = box.getBoundingClientRect();
    const c = Math.floor((e.clientX - rect.left) / rect.width * nCols);
    const r = Math.floor((e.clientY - rect.top) / rect.height * nRows);
    if (r < 0 || r >= nRows || c < 0 || c >= nCols) {{ tip.style.display = "none"; return; }}
    const v = d.values[r][c];
    tip.textContent = d.ids[r] + " · " + d.lobes[c] + ": " + (v === null ? "missing" : v.toFixed(3));
    tip.style.left = (e.clientX - outer.left + 12) + "px"; tip.style.top = (e.clientY - outer.top + 12) + "px";
    tip.style.display = "block";
  }});
  canvas.addEventListener("mouseleave", function() {{ tip.style.display = "none"; }});
}})();
</script>"""
