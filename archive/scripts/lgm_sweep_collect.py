#!/usr/bin/env python3
"""Aggregate LGM sweep results into a ranked table.

Reads each config dir under experiments/lgm_sweep/<TAG>/:
  - genuine_<TAG>.json            -> prediction: accuracy, perplexity
  - stylized_facts/stylized_facts_<TAG>.json -> simulation: Fano/ACF facts (real vs model)
  - master.log                    -> closed_form_rho (branching), config echo

Simulation score = mean relative error of model vs real on a few headline facts
(F5 Fano at 1s, F6 |r|-ACF clustering, F2 heavy-tail kurtosis). Lower is better.
"""
import json, os, re, sys, glob, math

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/simulation/experiments/lgm_sweep")


def _relerr(model, real):
    if real is None or model is None:
        return float("nan")
    try:
        m, r = float(model), float(real)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(m) or not math.isfinite(r):
        return float("nan")
    denom = abs(r) if abs(r) > 1e-9 else 1e-9
    return abs(m - r) / denom


def load_config(base):
    log = os.path.join(base, "master.log")
    rho = None
    seq = stride = rhocap = M = hid = vfb = None
    if os.path.exists(log):
        txt = open(log).read()
        mr = re.search(r"closed_form_rho=([0-9.]+)", txt)
        if mr:
            rho = float(mr.group(1))
        ms = re.search(r"SEQ=(\S+) STRIDE=(\S+) RHO=(\S+) M=(\S+) HID=(\S+) VFB=(\S+)", txt)
        if ms:
            seq, stride, rhocap, M, hid, vfb = ms.groups()
    return rho, seq, stride, rhocap, M, hid, vfb


def main():
    rows = []
    for base in sorted(glob.glob(os.path.join(ROOT, "*"))):
        if not os.path.isdir(base):
            continue
        tag = os.path.basename(base)
        gj = glob.glob(os.path.join(base, "genuine_*.json"))
        sj = glob.glob(os.path.join(base, "stylized_facts", "stylized_facts_*.json"))
        acc = ppl = None
        if gj:
            g = json.load(open(gj[0]))
            acc = g.get("genuine_mark_accuracy")
            ppl = g.get("genuine_mark_perplexity")
        fano_re = clus_re = kurt_re = None
        sim_score = float("nan")
        if sj:
            s = json.load(open(sj[0]))
            h = s.get("headline", {})
            # F5 Fano across scales (clustering of event counts): mean rel err over scales
            f5 = h.get("F5 Fano at scales")
            if f5:
                errs = [_relerr(m, r) for m, r in zip(f5["model"], f5["real"])]
                errs = [e for e in errs if e == e]
                if errs:
                    fano_re = sum(errs) / len(errs)
            # F6 |r|-ACF lags1-10 (volatility clustering / long memory)
            f6 = h.get("F6 ACF|r| lags1-10 (>0)")
            if f6:
                clus_re = _relerr(f6["model"], f6["real"])
            # F2 excess kurtosis (heavy tails)
            f2 = h.get("F2 excess kurtosis (>0)")
            if f2:
                kurt_re = _relerr(f2["model"], f2["real"])
            parts = [x for x in (fano_re, clus_re, kurt_re) if x == x]  # drop nan
            if parts:
                sim_score = sum(parts) / len(parts)
        rho, seq, stride, rhocap, M, hid, vfb = load_config(base)
        rows.append(dict(tag=tag, seq=seq, stride=stride, rhocap=rhocap, M=M, hid=hid, vfb=vfb,
                         rho=rho, acc=acc, ppl=ppl, fano_re=fano_re, clus_re=clus_re,
                         kurt_re=kurt_re, sim=sim_score))

    def f(x, fmt="{:.4f}"):
        return fmt.format(x) if isinstance(x, (int, float)) and x == x else "-"

    print(f"\nLGM sweep results  ({ROOT})\n")
    hdr = ("TAG", "seq", "str", "rhoCap", "M", "hid", "vfb", "rho", "ACC↑", "PPL↓", "Fano_re", "clus_re", "kurt_re", "SIM↓")
    print("{:<17} {:>4} {:>4} {:>6} {:>2} {:>4} {:>3} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}".format(*hdr))
    for r in rows:
        print("{:<17} {:>4} {:>4} {:>6} {:>2} {:>4} {:>3} {:>6} {:>7} {:>8} {:>8} {:>8} {:>8} {:>7}".format(
            r["tag"], r["seq"] or "-", r["stride"] or "-", r["rhocap"] or "-", r["M"] or "-",
            r["hid"] or "-", r["vfb"] or "-", f(r["rho"]), f(r["acc"]), f(r["ppl"], "{:.3f}"),
            f(r["fano_re"]), f(r["clus_re"]), f(r["kurt_re"]), f(r["sim"])))

    done = [r for r in rows if isinstance(r["acc"], (int, float)) and r["acc"] == r["acc"]]
    if done:
        best_pred = max(done, key=lambda r: r["acc"])
        sim_done = [r for r in done if r["sim"] == r["sim"]]
        best_sim = min(sim_done, key=lambda r: r["sim"]) if sim_done else None
        print(f"\nBEST PREDICTION: {best_pred['tag']}  acc={f(best_pred['acc'])} ppl={f(best_pred['ppl'],'{:.3f}')}")
        if best_sim:
            print(f"BEST SIMULATION: {best_sim['tag']}  sim_relerr={f(best_sim['sim'])} rho={f(best_sim['rho'])}")
    print(f"\n{len(done)}/{len(rows)} configs have results.")


if __name__ == "__main__":
    main()
