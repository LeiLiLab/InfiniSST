#!/usr/bin/env python3
"""Draw a compact HTML/SVG retriever inference figure and export previews."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


OUT_DIR = Path(__file__).resolve().parent
BASE = "retriever_inference_maxsim_compact"
WIDTH = 1600
HEIGHT = 720
PAGE_WIDTH_IN = 13.333
PAGE_HEIGHT_IN = 6.0


def svg_markup() -> str:
    return f"""<svg id="figure" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Timeline-aware MaxSim retriever inference">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#263238"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
    <linearGradient id="heat" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#67b77a"/>
      <stop offset="55%" stop-color="#f0dd72"/>
      <stop offset="100%" stop-color="#f59e57"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#111827" flood-opacity="0.10"/>
    </filter>
  </defs>

  <style>
    .title {{ font: 700 44px Arial, Helvetica, sans-serif; fill: #111827; }}
    .subtitle {{ font: 400 23px Arial, Helvetica, sans-serif; fill: #4b5563; }}
    .section {{ font: 700 24px Arial, Helvetica, sans-serif; fill: #111827; }}
    .label {{ font: 400 20px Arial, Helvetica, sans-serif; fill: #111827; }}
    .small {{ font: 400 17px Arial, Helvetica, sans-serif; fill: #374151; }}
    .tiny {{ font: 400 15px Arial, Helvetica, sans-serif; fill: #4b5563; }}
    .math {{ font: italic 21px Georgia, 'Times New Roman', serif; fill: #111827; }}
    .num {{ font: 700 17px Arial, Helvetica, sans-serif; fill: #ffffff; }}
    .line {{ stroke: #263238; stroke-width: 3; fill: none; marker-end: url(#arrow); }}
    .thin {{ stroke: #263238; stroke-width: 2; fill: none; marker-end: url(#arrow); }}
    .blueLine {{ stroke: #2563eb; stroke-width: 3; fill: none; marker-end: url(#arrowBlue); }}
    .muted {{ stroke: #9ca3af; stroke-width: 2; fill: none; }}
    .dash {{ stroke-dasharray: 8 8; }}
  </style>

  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
  <text x="72" y="74" class="title">Timeline-Aware MaxSim Retrieval</text>

  <g transform="translate(82,140)">
    <circle cx="0" cy="-2" r="18" fill="#111827"/><text x="0" y="4" text-anchor="middle" class="num">1</text>
    <text x="32" y="5" class="section">Context</text>
    <text x="0" y="76" class="label">speech timeline</text>
    <line x1="0" y1="170" x2="300" y2="170" class="line"/>
    <rect x="96" y="119" width="104" height="102" rx="0" fill="#e5e7eb" stroke="#6b7280" stroke-width="2"/>
    <rect x="200" y="82" width="74" height="139" rx="0" fill="#16a7df" stroke="#087da8" stroke-width="2"/>
    <line x1="200" y1="60" x2="200" y2="252" class="muted dash"/>
    <text x="148" y="161" text-anchor="middle" class="label">1.92s</text>
    <text x="148" y="190" text-anchor="middle" class="small">look-back</text>
    <text x="237" y="149" text-anchor="middle" class="math" fill="#ffffff">cᵢ</text>
    <text x="237" y="181" text-anchor="middle" class="small" fill="#ffffff" style="font-weight:700">current</text>
    <text x="237" y="207" text-anchor="middle" class="small" fill="#ffffff" style="font-weight:700">chunk</text>
    <text x="32" y="203" text-anchor="middle" class="math">cᵢ₋₁</text>
    <text x="305" y="203" text-anchor="middle" class="math">cᵢ₊₁</text>
    <text x="230" y="45" class="tiny">step boundary</text>
    <text x="148" y="300" text-anchor="middle" class="math">aᵢ = look-back + cᵢ</text>
  </g>

  <path d="M 390 310 C 440 310, 455 310, 505 310" class="line"/>

  <g transform="translate(520,140)">
    <circle cx="0" cy="-2" r="18" fill="#111827"/><text x="0" y="4" text-anchor="middle" class="num">2</text>
    <text x="32" y="5" class="section">Multi-scale windows</text>
    <rect x="55" y="82" width="158" height="64" rx="10" fill="#c7edb6" stroke="#45833b" stroke-width="2"/>
    <text x="134" y="121" text-anchor="middle" class="label">speech encoder</text>
    <path d="M 134 148 L 134 196" class="thin"/>
    <circle cx="66" cy="226" r="25" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2"/>
    <circle cx="134" cy="226" r="25" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2"/>
    <circle cx="248" cy="226" r="25" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2"/>
    <text x="66" y="233" text-anchor="middle" class="math">h₁</text>
    <text x="134" y="233" text-anchor="middle" class="math">h₂</text>
    <text x="191" y="233" text-anchor="middle" class="section">...</text>
    <text x="248" y="233" text-anchor="middle" class="math">hₘ</text>
    <g opacity="0.95">
      <rect x="36" y="328" width="138" height="38" fill="#d7ebfb" stroke="#6f9ccf" stroke-width="2"/>
      <rect x="82" y="362" width="160" height="38" fill="#d7ebfb" stroke="#6f9ccf" stroke-width="2"/>
      <rect x="126" y="396" width="186" height="38" fill="#d7ebfb" stroke="#6f9ccf" stroke-width="2"/>
      <rect x="172" y="430" width="142" height="38" fill="#d7ebfb" stroke="#6f9ccf" stroke-width="2"/>
    </g>
    <path d="M 92 470 L 92 390" class="muted" marker-end="url(#arrow)"/>
    <text x="64" y="505" class="tiny">overlap cᵢ</text>
    <text x="170" y="528" text-anchor="middle" class="label">candidate windows zₘ</text>
  </g>

  <path d="M 850 310 C 900 310, 915 310, 965 310" class="line"/>
  <path d="M 834 535 C 895 535, 920 535, 965 535" class="line"/>

  <g transform="translate(985,140)">
    <circle cx="0" cy="-2" r="18" fill="#111827"/><text x="0" y="4" text-anchor="middle" class="num">3</text>
    <text x="32" y="5" class="section">MaxSim scoring</text>
    <text x="30" y="86" class="small">speech windows</text>
    <text x="215" y="86" class="small">glossary bank</text>
    <rect x="28" y="122" width="230" height="230" fill="url(#heat)" filter="url(#softShadow)"/>
    <g opacity="0.20" stroke="#ffffff" stroke-width="2">
      <line x1="74" y1="122" x2="74" y2="352"/><line x1="120" y1="122" x2="120" y2="352"/>
      <line x1="166" y1="122" x2="166" y2="352"/><line x1="212" y1="122" x2="212" y2="352"/>
      <line x1="28" y1="168" x2="258" y2="168"/><line x1="28" y1="214" x2="258" y2="214"/>
      <line x1="28" y1="260" x2="258" y2="260"/><line x1="28" y1="306" x2="258" y2="306"/>
    </g>
    <rect x="118" y="208" width="92" height="54" rx="8" fill="#c7edb6" stroke="#45833b" stroke-width="2"/>
    <text x="164" y="242" text-anchor="middle" class="label">max</text>
    <rect x="314" y="126" width="74" height="56" fill="#f5ad82" stroke="#b85f2f" stroke-width="2"/>
    <rect x="314" y="204" width="74" height="56" fill="#f5ad82" stroke="#b85f2f" stroke-width="2"/>
    <rect x="314" y="316" width="74" height="56" fill="#f5ad82" stroke="#b85f2f" stroke-width="2"/>
    <text x="351" y="162" text-anchor="middle" class="math">g₁</text>
    <text x="351" y="240" text-anchor="middle" class="math">g₂</text>
    <text x="351" y="300" text-anchor="middle" class="section">...</text>
    <text x="351" y="352" text-anchor="middle" class="math">gₙ</text>
    <path d="M 258 235 L 314 154" stroke="#b85f2f" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
    <path d="M 258 235 L 314 232" stroke="#b85f2f" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
    <path d="M 258 235 L 314 344" stroke="#b85f2f" stroke-width="2" fill="none" marker-end="url(#arrow)"/>
    <path d="M 314 154 L 258 212" stroke="#45833b" stroke-width="2" fill="none" marker-end="url(#arrowBlue)"/>
    <path d="M 314 232 L 258 232" stroke="#45833b" stroke-width="2" fill="none" marker-end="url(#arrowBlue)"/>
    <text x="190" y="440" text-anchor="middle" class="math">score(eⱼ) = maxₘ zₘᵀ gⱼ</text>
  </g>

  <path d="M 1390 310 C 1425 310, 1436 310, 1475 310" class="line"/>

  <g transform="translate(1245,140)">
    <circle cx="0" cy="-2" r="18" fill="#111827"/><text x="0" y="4" text-anchor="middle" class="num">4</text>
    <text x="32" y="5" class="section">Step-local map</text>
    <g transform="translate(128,92)">
      <rect x="0" y="0" width="176" height="46" rx="8" fill="#c7edb6" stroke="#45833b" stroke-width="2"/>
      <text x="88" y="30" text-anchor="middle" class="small">overlap cᵢ</text>
      <rect x="0" y="66" width="176" height="46" rx="8" fill="#c7edb6" stroke="#45833b" stroke-width="2"/>
      <text x="88" y="96" text-anchor="middle" class="small">score ≥ τ</text>
      <rect x="0" y="132" width="176" height="46" rx="8" fill="#c7edb6" stroke="#45833b" stroke-width="2"/>
      <text x="88" y="162" text-anchor="middle" class="small">top-K</text>
    </g>
    <path d="M 216 280 L 216 342" class="thin"/>
    <rect x="100" y="356" width="232" height="126" rx="10" fill="#f5ad82" stroke="#b85f2f" stroke-width="2"/>
    <text x="216" y="392" text-anchor="middle" class="label">Gᵢ</text>
    <text x="216" y="428" text-anchor="middle" class="small">source → target</text>
    <text x="216" y="458" text-anchor="middle" class="small">source → target</text>
    <text x="216" y="538" text-anchor="middle" class="math">stateless per step</text>
  </g>
</svg>"""


def html_markup(svg: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Timeline-Aware MaxSim Retriever</title>
  <style>
    @page {{ size: {PAGE_WIDTH_IN}in {PAGE_HEIGHT_IN}in; margin: 0; }}
    html, body {{
      margin: 0;
      width: 100%;
      min-height: 100%;
      background: #ffffff;
    }}
    body {{
      display: grid;
      place-items: center;
      font-family: Arial, Helvetica, sans-serif;
    }}
    .page {{
      width: {PAGE_WIDTH_IN}in;
      height: {PAGE_HEIGHT_IN}in;
      background: #ffffff;
    }}
    svg {{
      display: block;
      width: 100%;
      height: 100%;
    }}
  </style>
</head>
<body>
  <main class="page">
{svg}
  </main>
</body>
</html>
"""


def export_with_playwright(html_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2)
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(OUT_DIR / f"{BASE}.pdf"),
            width=f"{PAGE_WIDTH_IN}in",
            height=f"{PAGE_HEIGHT_IN}in",
            print_background=True,
            prefer_css_page_size=True,
        )
        page.screenshot(path=str(OUT_DIR / f"{BASE}.png"), full_page=True)
        browser.close()


def main() -> int:
    svg = svg_markup()
    html = html_markup(svg)
    svg_path = OUT_DIR / f"{BASE}.svg"
    html_path = OUT_DIR / f"{BASE}.html"
    svg_path.write_text(svg + "\n", encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")
    export_with_playwright(html_path)
    print(f"Wrote {html_path}")
    print(f"Wrote {svg_path}")
    print(f"Wrote {OUT_DIR / (BASE + '.pdf')}")
    print(f"Wrote {OUT_DIR / (BASE + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
