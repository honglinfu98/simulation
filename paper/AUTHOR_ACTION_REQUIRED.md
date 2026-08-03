# Author Action Required

Only items that could not be resolved from the project's code, logs, and data.
Each has a matching non-rendered `% AUTHOR ACTION REQUIRED` comment in the
LaTeX source where noted.

## 1. Faithful published-S2P2 baseline (new experiment)
- **Question.** Should a faithful port of the published per-type
  ScaledSoftplus decoder (complex DPLR states, pre-norm, GLU, impulse-input)
  be added as a baseline and re-run (9 arms)?
- **Why it matters.** The paper now honestly evaluates only the factorized
  uncapped ablation (S2P2-U). A reviewer may ask whether the published
  decoder verbatim also fails in closed loop.
- **Affects.** Introduction, Background, Experiments (baseline list),
  Conclusion.
- **Minimum needed.** 9 training arms + calibration/SF stages on the cluster
  (config knobs partially exist; complex-state port is new work).
- **Blocking?** No — the manuscript now explicitly scopes every claim to
  S2P2-U. Strongly recommended if time permits.

## 2. Head-component ablations (new experiments)
- **Question.** Bounded gate without cap; cap without gate; factorization
  without cap; full head.
- **Why.** Component-level causal attribution. The manuscript now presents
  the design-choice bullets as intent + empirical association ("the first
  three choices are intended to...", "coincided with...") and the conclusion
  says attribution "awaits ablations".
- **Blocking?** No, given the weakened wording. Needed before any future
  claim that a specific component causes calibratability.

## 3. Dataset documentation (information)
- **Question.** Exact collection dates, timezone, train/val/test boundaries,
  per-split event counts, number of genuine segments per asset, timestamp
  resolution of the Kaiko feed, and data-quality counters (unmatched trades,
  ambiguous matches, dropped updates, multi-event timestamps).
- **Where.** `% AUTHOR ACTION` comments in `section/method.tex` (data
  collector) and `section/appendix.tex` (Dataset).
- **Minimum needed.** One pass of `scripts/data_summary.py` (or equivalent)
  over the raw archives on the cluster; collection dates from the collector
  logs.
- **Blocking?** Should be resolved before camera-ready; not
  submission-blocking (test-event counts and split rates are already stated).

## 4. IS events deeper than 10 ticks (implementation question)
- **Question.** The constructor emits uncapped IS tick distances
  (`event_construction_chunked.py:338`), but the vocabulary holds IS levels
  1..10 only. Are deeper improvements clamped to 10, dropped, or did they
  never occur in these streams?
- **Where.** `% AUTHOR ACTION` comment in `section/method.tex` (event
  constructor).
- **Minimum needed.** Grep the channel-mapping step or count raw IS levels
  >10 in the built event files.
- **Blocking?** No, but the Methods text must not be finalized as
  "δ ∈ {1..10}" for IS without this check.

## 5. Speed-bench raw artifacts (evidence archiving)
- **Question.** Archive the raw logs behind Table 3 (train-step times,
  thinning acceptance 0.54, scan-equivalence check incl. the norm used for
  the 2×10⁻⁴ discrepancy), and confirm the PCT-LSTM row was measured with
  the current SD-PNHP backbone (the row predates the backbone swap in the
  session history).
- **Where.** `% AUTHOR ACTION` comment in `section/experiments.tex`
  (Efficiency).
- **Blocking?** No for submission; yes for the released artifact bundle.

## 6. Per-seed values in the supplement
- **Question.** The Statistics bullet promises indicative CIs; the appendix
  should carry per-seed raw values for Tables 1–2 (three numbers per cell).
- **Minimum needed.** One run of a variant of `make_tables.py` that emits
  per-seed tables (data already in `results.json`).
- **Blocking?** No; recommended.

## 7. Venue placement of the Technical Appendix
- **Question.** AAAI-26 may require the appendix as separate supplementary
  material rather than trailing the references in the main PDF.
- **Where.** `main.tex` comment above `\input{section/appendix}`.
- **Blocking?** Check the author kit before submission; moving the appendix
  out is a one-line change.

## 8. Kaiko citation access date
- **Question.** Venue-required access date for the Kaiko data-provider
  reference (entry currently year-only).
- **Blocking?** No.
