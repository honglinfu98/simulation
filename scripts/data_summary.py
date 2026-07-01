#!/usr/bin/env python3
"""Self-contained event-data summary stats + plots (no repo deps).

    python data_summary.py --label gmni_eth --out /tmp/ds <file.jsonl.gz> [more files...]

Reads event-set JSONL (one set per line: {timestamp_ns, event_count, events:[{event_type,
side, level, volume, price}], lob_state}). Computes per-dataset: event/set rate,
set-size (simultaneity) distribution, event-type composition, inter-arrival times,
volume distribution. Writes <label>_summary.json and <label>_panel.png to --out.
"""
import argparse, gzip, io, json, os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _open(p):
    return io.TextIOWrapper(gzip.open(p, "rb")) if p.endswith(".gz") else open(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lines", type=int, default=0, help="0 = all")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    n_sets = n_events = 0
    setsize = Counter()
    by_type = Counter()       # MO/CO/LO/IS
    by_typeside = Counter()
    by_level = Counter()
    ts_first = ts_last = None
    dts = []                  # inter-set gaps (s)
    vols = []
    prev_ts = None
    for fp in a.files:
        with _open(fp) as fh:
            for line in fh:
                if a.max_lines and n_sets >= a.max_lines:
                    break
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                ts = d.get("timestamp")
                if ts is None:
                    continue
                ec = int(d.get("event_count", len(d.get("events", []))))
                n_sets += 1
                n_events += ec
                setsize[min(ec, 5)] += 1
                if ts_first is None:
                    ts_first = ts
                ts_last = ts
                if prev_ts is not None:
                    dt = (ts - prev_ts) / 1e9
                    if 0 <= dt < 60:
                        dts.append(dt)
                prev_ts = ts
                for e in d.get("events", []):
                    t = e.get("event_type", "?")
                    s = e.get("side", "?")
                    by_type[t] += 1
                    by_typeside[f"{t}_{s}"] += 1
                    by_level[min(int(e.get("level", 0)), 10)] += 1
                    v = e.get("volume")
                    if v and len(vols) < 200000:
                        vols.append(float(v))

    span_s = (ts_last - ts_first) / 1e9 if (ts_first and ts_last and ts_last > ts_first) else 0.0
    singleton_frac = setsize[1] / max(n_sets, 1)
    dts = np.array(dts) if dts else np.array([0.0])
    vols = np.array(vols) if vols else np.array([0.0])
    summary = {
        "label": a.label, "files": len(a.files),
        "n_event_sets": n_sets, "n_events": n_events,
        "span_seconds": round(span_s, 1), "span_hours": round(span_s / 3600, 2),
        "set_rate_per_s": round(n_sets / span_s, 4) if span_s else None,
        "event_rate_per_s": round(n_events / span_s, 4) if span_s else None,
        "mean_set_size": round(n_events / max(n_sets, 1), 4),
        "singleton_fraction": round(singleton_frac, 5),
        "simultaneity_rate": round(1 - singleton_frac, 5),
        "set_size_hist": {str(k): setsize[k] for k in sorted(setsize)},
        "by_event_type": dict(by_type.most_common()),
        "by_type_side": dict(by_typeside.most_common()),
        "dt_median_s": round(float(np.median(dts)), 5),
        "dt_frac_eq_0.1s": round(float(np.mean(np.abs(dts - 0.1) < 1e-4)), 4),
        "volume_median": round(float(np.median(vols)), 6),
    }
    jp = os.path.join(a.out, f"{a.label}_summary.json")
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2)

    # panel of 4 plots
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(f"{a.label}  |  {n_events:,} events / {n_sets:,} sets over {span_s/3600:.1f} h "
                 f"|  rate {summary['event_rate_per_s']} ev/s  |  singleton {singleton_frac:.3%}", fontsize=11)
    ks = sorted(setsize)
    ax[0,0].bar([str(k) if k < 5 else "5+" for k in ks], [setsize[k] for k in ks], color="#2b6cb0")
    ax[0,0].set_yscale("log"); ax[0,0].set_title("Set size (event_count) per arrival"); ax[0,0].set_xlabel("|x|")
    top = by_typeside.most_common(16)
    ax[0,1].barh([k for k,_ in top][::-1], [v for _,v in top][::-1], color="#2f855a")
    ax[0,1].set_title("Event type × side (top 16)"); ax[0,1].set_xscale("log")
    d = dts[dts > 0]
    if d.size:
        ax[1,0].hist(np.clip(d, 1e-4, 5), bins=np.logspace(-4, 0.7, 60), color="#4a5568")
        ax[1,0].set_xscale("log")
    ax[1,0].set_title(f"Inter-arrival dt (s)  median={summary['dt_median_s']}"); ax[1,0].set_xlabel("seconds")
    v = vols[vols > 0]
    if v.size:
        ax[1,1].hist(np.log10(v), bins=60, color="#b7791f")
    ax[1,1].set_title("log10 volume"); ax[1,1].set_xlabel("log10(volume)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    pp = os.path.join(a.out, f"{a.label}_panel.png")
    fig.savefig(pp, dpi=130); plt.close(fig)

    print(json.dumps(summary, indent=2))
    print("WROTE", jp, pp)


if __name__ == "__main__":
    main()
