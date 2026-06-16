"""Genuine-event prediction metrics for any VolumeSetMTPP checkpoint, computed
CONSISTENTLY so models with different mark heads are comparable.

The gmni-marks data is one-mark-or-empty per slot (~33% empty, 0% multi).  The
comparable mark metric is therefore over GENUINE events only:
  - top-1 mark accuracy: argmax(mark logits) == true mark, on non-empty targets
  - mark perplexity: exp(mean -log p(true mark)) over non-empty targets,
    where p is softmax(logits) for BOTH bernoulli and categorical heads (the
    mark *ranking* is what we score, head-agnostic).
Empty-target positions are excluded identically for every model, so the number
is head-agnostic and fair across the whole comparison.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from volume_set_mtpp.training_evaluation.bfnx_data_loader import create_bfnx_dataloaders
from volume_set_mtpp.models.volume_set_mtpp import create_volume_set_mtpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--max-files", type=int, default=7)
    ap.add_argument("--seq-length", type=int, default=50)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--label", default="model")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dl = dict(cache_dir=args.cache_dir) if args.cache_dir else {}
    _, _, test_loader, em = create_bfnx_dataloaders(
        args.data_dir, args.batch_size, args.seq_length, args.stride, args.max_files, num_workers=0, **dl)

    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = create_volume_set_mtpp(em.num_events, cfg, device,
                                   use_volume=cfg.get("use_volume", True),
                                   intensity_type=cfg.get("intensity_type", "dynamic"))
    model.load_state_dict(ck["model_state_dict"]); model.to(device); model.eval()

    n_ev = 0
    correct = 0
    nll_sum = 0.0
    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            if bi >= args.max_batches:
                break
            im = batch["input_marks"].to(device).float()
            it = batch["input_times"].to(device).float()
            tm = batch["target_marks"].to(device).float()
            tt = batch["target_time"].to(device).float()
            ts = torch.cumsum(it, dim=1)
            states = model.decoder.get_states(im, ts)
            q = ts[:, -1:] + tt.unsqueeze(1).clamp_min(0.0)
            h = model.decoder.get_hidden_h(state_values=states, state_times=ts, timestamps=q)
            d = model.get_total_intensity_and_items(h)
            logits = d["item_logits"].squeeze(1)              # [B,K]
            ev = tm.sum(dim=1) > 0                            # genuine-event mask
            if ev.sum() == 0:
                continue
            tgt = tm[ev].argmax(dim=1)
            lg = logits[ev]
            pred = lg.argmax(dim=1)
            correct += (pred == tgt).sum().item()
            nll = F.cross_entropy(lg, tgt, reduction="sum").item()   # ranking-based, head-agnostic
            nll_sum += nll
            n_ev += int(ev.sum().item())

    acc = correct / max(n_ev, 1)
    ppl = math.exp(nll_sum / max(n_ev, 1))
    res = {"label": args.label, "checkpoint": args.checkpoint, "n_genuine_events": n_ev,
           "genuine_mark_accuracy": acc, "genuine_mark_perplexity": ppl,
           "mark_head": cfg.get("mark_head", "bernoulli")}
    print(json.dumps(res, indent=2), flush=True)
    if args.output:
        Path(args.output).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
