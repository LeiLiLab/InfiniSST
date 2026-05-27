# Retriever Inference MaxSim Draft

- Purpose: editable draft for an inference-time retriever mechanism figure.
- Paper label: not assigned yet.
- LaTeX source: not inserted yet.
- Preferred editable source: `retriever_inference_maxsim_clean.svg`
- Preferred paper output: `retriever_inference_maxsim_clean.pdf`
- Preview output: `retriever_inference_maxsim_clean.png`
- Preferred generator: `draw_retriever_inference_maxsim_svg.py`
- Older HTML draft: `retriever_inference_maxsim_compact.html`
- Older PPT draft: `retriever_inference_maxsim_draft.pptx`

The clean SVG draft focuses on the online retriever only: look-back context,
semi-transparent candidate speech windows, MaxSim scoring against the glossary
bank, compact filtering, and a concrete `Term_map` output. It intentionally
avoids framed panels and step headings.

Regenerate:

```bash
python draw_retriever_inference_maxsim_svg.py
```

The SVG is the source of truth for layout/style edits. The PDF is exported by
Playwright from the SVG source.
