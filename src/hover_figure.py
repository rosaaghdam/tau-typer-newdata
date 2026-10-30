"""A matplotlib figure on the page with a label under the mouse for some of its dots: the picture is unchanged;
invisible round spots are laid over the dots and carry the labels. No extra package is needed."""
import base64
import html
import io
import json

import matplotlib.pyplot as plt


def hover_figure_html(fig, ax, xy, tips, marked=None, element_id="tt_hover", dpi=192):
    """fig/ax: the figure and the axes the dots are in; xy: the dots' data coordinates (n x 2); tips: one label per
    dot; marked: True for dots to ring in red (e.g. a wrong subtype). The picture is shown at its real size (as the
    app's other figures) and shrinks with a narrow page."""
    fig.canvas.draw()
    where = fig.transFigure.inverted().transform(ax.transData.transform(xy))   # 0-1 of the figure, any screen
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)                     # not "tight": the dots must keep their place
    plt.close(fig)
    picture = base64.b64encode(buffer.getvalue()).decode()
    spots = []
    for (x, y), tip, mark in zip(where, tips, marked if marked is not None else [False] * len(tips)):
        ring = "border:2px solid #C62828;" if mark else ""
        spots.append(f'<span data-tip="{html.escape(tip)}" style="position:absolute; left:{100 * x:.3f}%; '
                     f'top:{100 * (1 - y):.3f}%; width:14px; height:14px; margin:-7px 0 0 -7px; '
                     f'border-radius:50%; cursor:crosshair; {ring}"></span>')
    return f"""
<div id="{element_id}" style="position:relative; width:100%; max-width:{fig.get_figwidth() * 96:.0f}px;">
  <img src="data:image/png;base64,{picture}" style="width:100%; display:block;" alt="figure">
  {"".join(spots)}
  <div class="tt-dot-tip" style="position:absolute; display:none; pointer-events:none; background:#1F2933; color:#fff;
    font-size:12px; padding:4px 8px; border-radius:4px; white-space:nowrap; z-index:10;"></div>
</div>
<script>
(function() {{
  const box = document.getElementById({json.dumps(element_id)}), tip = box.querySelector(".tt-dot-tip");
  box.addEventListener("mousemove", function(e) {{
    const spot = e.target.closest("span[data-tip]");
    if (!spot) {{ tip.style.display = "none"; return; }}
    const outer = box.getBoundingClientRect();
    tip.textContent = spot.dataset.tip;
    tip.style.display = "block";
    let x = e.clientX - outer.left + 12;                       // keep the label inside the figure
    if (x + tip.offsetWidth > outer.width) x = e.clientX - outer.left - tip.offsetWidth - 12;
    tip.style.left = Math.max(0, x) + "px"; tip.style.top = (e.clientY - outer.top + 12) + "px";
  }});
  box.addEventListener("mouseleave", function() {{ tip.style.display = "none"; }});
}})();
</script>"""
