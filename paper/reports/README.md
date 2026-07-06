# paper/reports/ — standalone technical reports (committed PDFs alongside source)

Self-contained documents (each has its own `\documentclass`) written during the study;
the paper (`../main.tex`) is distilled from them. Compile from `paper/` so relative
`figs/` paths resolve: `latexmk -pdf reports/<name>.tex`.

| report | content |
|---|---|
| `model_comparison_report` | **Source of truth for the benchmark**: corrected 7-model prediction + simulation tables and the SS2P2 ablation chain (mirrored in `docs/RESULTS.md`). |
| `s2p2_vs_ss2p2` | Side-by-side of the two head designs on the shared backbone. |
| `ss2p2_formulation` / `ss2p2_spec` / `ss2p2_oneliner` | The SS2P2 design write-ups (full derivation / spec / one-pager). |
| `ss2p2_s2p2_results` | Earlier SS2P2-vs-S2P2 pair results with stylized-facts figures. |
| `eval_report` | The first 7-model evaluation report (superseded by `model_comparison_report`). |
