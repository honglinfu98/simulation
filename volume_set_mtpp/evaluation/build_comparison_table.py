"""Assemble the event-driven Gemini-ETH model comparison table across every arm
(baselines, Compound Hawkes, s2p2 variants, NMH) from their stylized-facts JSON
and genuine-event accuracy.  Emits a markdown table + a JSON dump.

Each stylized_facts_*.json carries facts_real (identical across arms, the target)
and facts_model.  Genuine accuracy comes from the json's own field when present
(baselines), a sibling genuine_*.json (NMH / ours), or a logged CHP_ACCURACY line
(Compound Hawkes).  Run on the cluster where all experiment dirs live.
"""
from __future__ import annotations
import argparse, glob, json, math, os, re
from pathlib import Path

E = os.path.expanduser("~/volume-set-mtpp/experiments")

# label -> (stylized-facts json glob, accuracy source, rho source)
# rho source: ("none",) | ("chp",) [json field branching_ratio_rho] | ("nmhlog", base_glob)
MODELS = [
    ("LSTM",            f"{E}/basesim_lstm_*/stylized_facts_lstm.json",          ("field",),   ("none",)),
    ("SAHP",            f"{E}/basesim_sahp_*/stylized_facts_sahp.json",          ("field",),   ("none",)),
    ("CT-LSTM",         f"{E}/basesim_ct-lstm_*/stylized_facts_ct-lstm.json",    ("field",),   ("none",)),
    ("PCT-LSTM",        f"{E}/basesim_pct-lstm_*/stylized_facts_pct-lstm.json",  ("field",),   ("none",)),
    ("CompoundHawkes",  f"{E}/chp_full/stylized_facts_compound_hawkes.json",     ("chp_log", f"{E}/chp_full"), ("chp",)),
    ("s2p2-cat",        f"{E}/gmni_marks_cat_*/stylized_facts/stylized_facts_s2p2m_cat.json", ("genuine_json", f"{E}/gmni_marks_cat_*/genuine_cat.json"), ("none",)),
    ("s2p2-pfa",        f"{E}/gmni_marks_pfa_*/stylized_facts/stylized_facts_s2p2m_pfa.json", ("genuine_json", f"{E}/gmni_marks_pfa_*/genuine_pfa.json"), ("none",)),
    ("NMH-MLE(uncon)",  f"{E}/gmni_marks_nmh_*/stylized_facts/stylized_facts_nmh.json",   ("genuine_json", f"{E}/gmni_marks_nmh_*/genuine_nmh.json"),  ("nmhlog", f"{E}/gmni_marks_nmh_*")),
    ("NMH(constrained)",f"{E}/gmni_marks_nmhc_*/stylized_facts/stylized_facts_nmhc.json", ("genuine_json", f"{E}/gmni_marks_nmhc_*/genuine_nmh.json"), ("nmhlog", f"{E}/gmni_marks_nmhc_*")),
    ("MT-Hawkes(rho.8)", f"{E}/mt_hawkes/stylized_facts_mt_hawkes.json", ("field",), ("chp",)),
]


def newest(pat):
    fs = sorted(glob.glob(pat))
    return fs[-1] if fs else None


def grep_acc_chp(d):
    for lf in glob.glob(f"{d}/*.log") + glob.glob(f"{d}/../*.log"):
        try:
            t = Path(lf).read_text()
        except Exception:
            continue
        m = re.search(r"CHP_ACCURACY genuine_acc=([\d.]+) perplexity=([\d.]+)", t)
        if m:
            return float(m.group(1)), float(m.group(2))
    return None, None


def fano1(f):
    v = f.get("f5_fano_vs_scale")
    return v[0] if v else float("nan")


def get_rho(src, sf_json_dict):
    kind = src[0]
    if kind == "chp":
        return sf_json_dict.get("branching_ratio_rho")
    if kind == "nmhlog":
        for b in sorted(glob.glob(src[1])):
            for lf in glob.glob(f"{b}/master.log"):
                m = re.search(r"closed_form_rho=([\d.]+)", Path(lf).read_text())
                if m:
                    return float(m.group(1))
    return None


def row(label, fm):
    return {
        "model": label,
        "fano_1s": fano1(fm),
        "f2_kurt": fm.get("f2_excess_kurtosis"),
        "f3_skew": fm.get("f3_skewness"),
        "f6_acf_abs": fm.get("f6_mean_acf_abs_1_10"),
        "f8_powerlaw": fm.get("f8_powerlaw_exponent"),
        "f9_leverage": fm.get("f9_mean_leverage_1_10"),
        "f11_asym": fm.get("f11_timescale_asymmetry"),
        "n_buckets": fm.get("n_buckets"),
    }


def main():
    rows = []
    real = None
    for label, pat, acc_src, rho_src in MODELS:
        sf = newest(pat)
        if not sf:
            print(f"# MISSING {label}: {pat}")
            continue
        d = json.load(open(sf))
        if real is None and "facts_real" in d:
            real = d["facts_real"]
        r = row(label, d["facts_model"])
        # accuracy
        acc = ppl = None
        kind = acc_src[0]
        if kind == "field":
            acc = d.get("genuine_mark_accuracy"); ppl = d.get("genuine_mark_perplexity")
        elif kind == "chp_log":
            acc, ppl = grep_acc_chp(acc_src[1])
        elif kind == "genuine_json":
            gj = newest(acc_src[1])
            if gj:
                g = json.load(open(gj)); acc = g.get("genuine_mark_accuracy"); ppl = g.get("genuine_mark_perplexity")
        r["acc"] = acc; r["ppl"] = ppl
        r["rho"] = get_rho(rho_src, d)
        rows.append(r)
    if real is not None:
        rr = row("REAL", real); rr["acc"] = 1.0; rr["ppl"] = None; rr["rho"] = None
        rows = [rr] + rows

    # markdown
    def fmt(x, p=2):
        return "" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{p}f}"
    hdr = ["Model", "GenAcc", "Ppl", "rho", "Fano(1s)", "F2 kurt", "F3 skew", "F6 |r|ACF", "F8 plaw", "F9 lev", "F11 asym"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        print("| " + " | ".join([
            r["model"], fmt(r.get("acc"), 3), fmt(r.get("ppl"), 1), fmt(r.get("rho"), 3), fmt(r["fano_1s"], 2),
            fmt(r["f2_kurt"], 1), fmt(r["f3_skew"], 2), fmt(r["f6_acf_abs"], 3),
            fmt(r["f8_powerlaw"], 2), fmt(r["f9_leverage"], 3), fmt(r["f11_asym"], 2),
        ]) + " |")
    Path("comparison_table.json").write_text(json.dumps(rows, indent=2))
    print("\n# wrote comparison_table.json")


if __name__ == "__main__":
    main()
