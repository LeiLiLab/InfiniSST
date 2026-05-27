#!/usr/bin/env python3
"""Draw a clean SVG retriever inference figure and export PDF/PNG previews."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright


OUT_DIR = Path(__file__).resolve().parent
BASE = "retriever_inference_maxsim_clean"
WIDTH = 1600
HEIGHT = 560
PAGE_WIDTH_IN = 13.333
PAGE_HEIGHT_IN = 4.667


def svg_markup() -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-label="Timeline-aware MaxSim retrieval">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#263238"/>
    </marker>
    <marker id="arrowBlue" viewBox="0 0 10 10" refX="8.4" refY="5" markerWidth="6.2" markerHeight="6.2" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
    <linearGradient id="heat" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#62b77b"/>
      <stop offset="55%" stop-color="#f0db6f"/>
      <stop offset="100%" stop-color="#f49b56"/>
    </linearGradient>
    <filter id="shadow" x="-15%" y="-15%" width="130%" height="130%">
      <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#111827" flood-opacity="0.12"/>
    </filter>
  </defs>

  <style>
    .label {{ font: 500 25px Arial, Helvetica, sans-serif; fill: #111827; }}
    .small {{ font: 400 20px Arial, Helvetica, sans-serif; fill: #374151; }}
    .tiny {{ font: 400 17px Arial, Helvetica, sans-serif; fill: #4b5563; }}
    .math {{ font: italic 30px Georgia, 'Times New Roman', serif; fill: #111827; }}
    .mathSmall {{ font: italic 25px Georgia, 'Times New Roman', serif; fill: #111827; }}
    .whiteBold {{ font: 700 20px Arial, Helvetica, sans-serif; fill: #ffffff; }}
    .term {{ font: 500 26px Arial, Helvetica, sans-serif; fill: #111827; }}
    .filterText {{ font: 500 22px Arial, Helvetica, sans-serif; fill: #111827; }}
    .line {{ stroke: #263238; stroke-width: 3.0; fill: none; marker-end: url(#arrow); }}
    .thin {{ stroke: #263238; stroke-width: 2.3; fill: none; marker-end: url(#arrow); }}
    .muted {{ stroke: #9ca3af; stroke-width: 2.4; fill: none; }}
    .dash {{ stroke-dasharray: 8 8; }}
  </style>

  <rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>

  <g transform="translate(45,55)">
    <text x="150" y="16" text-anchor="middle" class="label">speech timeline</text>
    <line x1="0" y1="152" x2="315" y2="152" class="line"/>
    <rect x="104" y="96" width="118" height="112" fill="#e5e7eb" stroke="#6b7280" stroke-width="2.6"/>
    <rect x="222" y="58" width="86" height="150" fill="#16a7df" stroke="#087da8" stroke-width="2.6"/>
    <line x1="222" y1="32" x2="222" y2="240" class="muted dash"/>
    <text x="163" y="144" text-anchor="middle" class="label">1.92s</text>
    <text x="163" y="176" text-anchor="middle" class="small">look-back</text>
    <text x="265" y="114" text-anchor="middle" class="math" fill="#ffffff">cᵢ</text>
    <text x="265" y="150" text-anchor="middle" class="whiteBold">current</text>
    <text x="265" y="179" text-anchor="middle" class="whiteBold">chunk</text>
    <text x="44" y="188" text-anchor="middle" class="mathSmall">cᵢ₋₁</text>
    <text x="326" y="188" text-anchor="middle" class="mathSmall">cᵢ₊₁</text>
    <text x="176" y="290" text-anchor="middle" class="math">aᵢ = cᵢ + look-back</text>
  </g>

  <path d="M 380 207 C 392 207, 408 207, 425 207" class="line"/>

  <g transform="translate(425,40)">
    <rect x="62" y="56" width="176" height="68" rx="12" fill="#c7edb6" stroke="#45833b" stroke-width="2.4"/>
    <text x="150" y="98" text-anchor="middle" class="label">speech encoder</text>
    <path d="M 150 126 L 150 180" class="thin"/>
    <circle cx="70" cy="212" r="28" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2.4"/>
    <circle cx="150" cy="212" r="28" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2.4"/>
    <circle cx="290" cy="212" r="28" fill="#b8dcff" stroke="#4f8cc9" stroke-width="2.4"/>
    <text x="70" y="222" text-anchor="middle" class="mathSmall">h₁</text>
    <text x="150" y="222" text-anchor="middle" class="mathSmall">h₂</text>
    <text x="220" y="222" text-anchor="middle" class="label">...</text>
    <text x="290" y="222" text-anchor="middle" class="mathSmall">hₘ</text>

    <g>
      <rect x="48" y="310" width="188" height="48" fill="#8fc8ed" fill-opacity="0.30" stroke="#3f7fbd" stroke-opacity="0.82" stroke-width="2.2"/>
      <rect x="88" y="337" width="216" height="48" fill="#8fc8ed" fill-opacity="0.34" stroke="#3f7fbd" stroke-opacity="0.82" stroke-width="2.2"/>
      <rect x="128" y="364" width="244" height="48" fill="#8fc8ed" fill-opacity="0.38" stroke="#3f7fbd" stroke-opacity="0.82" stroke-width="2.2"/>
      <rect x="168" y="391" width="230" height="48" fill="#8fc8ed" fill-opacity="0.42" stroke="#3f7fbd" stroke-opacity="0.82" stroke-width="2.2"/>
      <rect x="208" y="418" width="176" height="48" fill="#8fc8ed" fill-opacity="0.46" stroke="#3f7fbd" stroke-opacity="0.82" stroke-width="2.2"/>
    </g>
    <path d="M 108 480 L 108 365" class="muted" marker-end="url(#arrow)"/>
    <text x="78" y="478" class="tiny">overlap cᵢ</text>
    <text x="235" y="503" text-anchor="middle" class="mathSmall">candidate windows zₘ</text>
  </g>

  <path d="M 770 207 C 792 207, 818 207, 845 207" class="line"/>
  <path d="M 824 462 C 830 462, 838 462, 845 462" class="line"/>

  <g transform="translate(810,45)">
    <text x="130" y="25" text-anchor="middle" class="small">MaxSim</text>
    <text x="48" y="60" class="tiny">zₘ</text>
    <text x="250" y="60" class="tiny">gⱼ</text>
    <rect x="38" y="80" width="248" height="248" fill="url(#heat)" filter="url(#shadow)"/>
    <g opacity="0.22" stroke="#ffffff" stroke-width="2.2">
      <line x1="88" y1="80" x2="88" y2="328"/><line x1="138" y1="80" x2="138" y2="328"/>
      <line x1="188" y1="80" x2="188" y2="328"/><line x1="238" y1="80" x2="238" y2="328"/>
      <line x1="38" y1="130" x2="286" y2="130"/><line x1="38" y1="180" x2="286" y2="180"/>
      <line x1="38" y1="230" x2="286" y2="230"/><line x1="38" y1="280" x2="286" y2="280"/>
    </g>
    <rect x="142" y="173" width="96" height="58" rx="9" fill="#c7edb6" stroke="#45833b" stroke-width="2.4"/>
    <text x="190" y="211" text-anchor="middle" class="label">max</text>

    <rect x="340" y="88" width="84" height="60" fill="#f5ad82" stroke="#b85f2f" stroke-width="2.4"/>
    <rect x="340" y="178" width="84" height="60" fill="#f5ad82" stroke="#b85f2f" stroke-width="2.4"/>
    <rect x="340" y="306" width="84" height="60" fill="#f5ad82" stroke="#b85f2f" stroke-width="2.4"/>
    <text x="382" y="127" text-anchor="middle" class="mathSmall">g₁</text>
    <text x="382" y="217" text-anchor="middle" class="mathSmall">g₂</text>
    <text x="382" y="281" text-anchor="middle" class="label">...</text>
    <text x="382" y="345" text-anchor="middle" class="mathSmall">gₙ</text>

    <path d="M 286 202 L 340 118" stroke="#b85f2f" stroke-width="2.3" fill="none" marker-end="url(#arrow)"/>
    <path d="M 286 202 L 340 208" stroke="#b85f2f" stroke-width="2.3" fill="none" marker-end="url(#arrow)"/>
    <path d="M 286 202 L 340 336" stroke="#b85f2f" stroke-width="2.3" fill="none" marker-end="url(#arrow)"/>
    <path d="M 340 118 L 286 178" stroke="#2563eb" stroke-width="2.5" fill="none" marker-end="url(#arrowBlue)"/>
    <path d="M 340 208 L 286 205" stroke="#2563eb" stroke-width="2.5" fill="none" marker-end="url(#arrowBlue)"/>
    <text x="220" y="442" text-anchor="middle" class="math">score(eⱼ) = maxₘ zₘᵀgⱼ</text>
  </g>

  <path d="M 1238 213 C 1252 213, 1260 213, 1270 213" class="line"/>

  <g transform="translate(1165,45)">
    <rect x="105" y="86" width="315" height="90" rx="10" fill="#c7edb6" stroke="#45833b" stroke-width="2.6"/>
    <text x="262" y="120" text-anchor="middle" class="label">Filter</text>
    <text x="262" y="157" text-anchor="middle" class="filterText">overlap cᵢ ∧ score ≥ τ ∧ top-K</text>
    <path d="M 262 180 L 262 224" class="thin"/>
    <rect x="75" y="236" width="350" height="186" rx="12" fill="#f5ad82" stroke="#b85f2f" stroke-width="2.6"/>
    <text x="250" y="282" text-anchor="middle" class="term">TERM_MAP Gᵢ</text>
    <text x="250" y="334" text-anchor="middle" class="term">Shinzo Abe = 安倍晋三</text>
    <text x="250" y="378" text-anchor="middle" class="term">PM = 首相</text>
  </g>
</svg>"""


def export_with_playwright(svg_path: Path) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=2)
        page.goto(svg_path.resolve().as_uri(), wait_until="load")
        page.pdf(
            path=str(OUT_DIR / f"{BASE}.pdf"),
            width=f"{PAGE_WIDTH_IN}in",
            height=f"{PAGE_HEIGHT_IN}in",
            print_background=True,
        )
        page.screenshot(
            path=str(OUT_DIR / f"{BASE}.png"),
            clip={"x": 0, "y": 0, "width": WIDTH, "height": HEIGHT},
        )
        browser.close()


def main() -> int:
    svg_path = OUT_DIR / f"{BASE}.svg"
    svg_path.write_text(svg_markup() + "\n", encoding="utf-8")
    export_with_playwright(svg_path)
    print(f"Wrote {svg_path}")
    print(f"Wrote {OUT_DIR / (BASE + '.pdf')}")
    print(f"Wrote {OUT_DIR / (BASE + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
