"""The false-positive check's dot plot as a web element: the same picture as figures_apply.plot_cleanup, plus the
person under the mouse. Drawn by the browser as SVG; no extra package is needed."""
import html
import json

import numpy as np

from src.figures_apply import CLEANUP_GROUPS, CLEANUP_YLABEL, cleanup_jitter, short_region


def _hex(colour):
    """A matplotlib colour (name, hex or RGB) -> "#rrggbb"."""
    from matplotlib.colors import to_hex
    return to_hex(colour)


def cleanup_scatter_html(p, threshold, groups, element_id="ap_cleanup_svg", width=1000, height=420):
    """p: people x cleanup regions (probabilities 0-1); groups: each person's key in CLEANUP_GROUPS."""
    left, right, top, bottom = 95, 20, 20, 50                     # room for the axes (SVG units)
    band = (width - left - right) / p.shape[1]                    # one band per region
    y = lambda v: top + (1.03 - v) / 1.06 * (height - top - bottom)
    jitter = cleanup_jitter(len(p), p.shape[1])
    groups = groups.reindex(p.index).fillna("kept")
    parts = [f'<line x1="{left}" x2="{width - right}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="#E4E7EB"/>'
             f'<text x="{left - 8}" y="{y(t) + 4:.1f}" text-anchor="end">{t:.1f}</text>' for t in np.arange(0, 1.01, 0.2)]
    parts += [f'<text x="{left + band * (i + 0.5):.1f}" y="{height - 14}" text-anchor="middle">'
              f'{html.escape(short_region(roi))}</text>' for i, roi in enumerate(p.columns)]
    for key, (label, shade) in CLEANUP_GROUPS.items():           # S0 first, so the others lie on top
        rows = np.flatnonzero((groups == key).to_numpy())
        for i, roi in enumerate(p.columns):
            for r in rows:
                tip = f"{p.index[r]} · {short_region(roi)}: {p.iat[r, i]:.3f} · {label}"
                parts.append(f'<circle cx="{left + band * (i + 0.5 + jitter[r, i]):.1f}" cy="{y(p.iat[r, i]):.1f}" '
                             f'r="5" fill="{_hex(shade)}" fill-opacity="0.65" data-tip="{html.escape(tip)}"/>')
    parts.append(f'<line x1="{left}" x2="{width - right}" y1="{y(threshold):.1f}" y2="{y(threshold):.1f}" '
                 f'stroke="#C62828" stroke-width="1.5" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{left - 70}" y="{height / 2}" transform="rotate(-90 {left - 70} {height / 2})" '
                 f'text-anchor="middle">{CLEANUP_YLABEL}</text>')
    counts = groups.value_counts()
    legend = "".join(f'<span style="display:inline-flex; align-items:center; gap:5px; margin-right:16px;">'
                     f'<span style="width:10px; height:10px; border-radius:50%; background:{_hex(shade)};"></span>'
                     f'{html.escape(label)} (n = {int(counts.get(key, 0))})</span>'
                     for key, (label, shade) in CLEANUP_GROUPS.items())
    legend += (f'<span style="display:inline-flex; align-items:center; gap:5px;"><span style="width:18px; '
               f'border-top:2px dashed #C62828;"></span>threshold {threshold:g}</span>')
    return f"""
<div id="{element_id}" style="position:relative; font-size:13px; color:#52606D;">
  <div style="margin-bottom:6px;">{legend}</div>
  <svg viewBox="0 0 {width} {height}" style="width:100%; height:auto; display:block; font-size:22px; fill:#52606D;">
    {"".join(parts)}
  </svg>
  <div class="tt-dot-tip" style="position:absolute; display:none; pointer-events:none; background:#1F2933; color:#fff;
    font-size:12px; padding:4px 8px; border-radius:4px; white-space:nowrap; z-index:10;"></div>
</div>
<script>
(function() {{
  const box = document.getElementById({json.dumps(element_id)}), tip = box.querySelector(".tt-dot-tip");
  box.querySelector("svg").addEventListener("mousemove", function(e) {{
    const dot = e.target.closest("circle");
    if (!dot) {{ tip.style.display = "none"; return; }}
    const outer = box.getBoundingClientRect();
    tip.textContent = dot.dataset.tip;
    tip.style.left = (e.clientX - outer.left + 12) + "px"; tip.style.top = (e.clientY - outer.top + 12) + "px";
    tip.style.display = "block";
  }});
  box.querySelector("svg").addEventListener("mouseleave", function() {{ tip.style.display = "none"; }});
}})();
</script>"""
