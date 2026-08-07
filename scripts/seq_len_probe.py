#!/usr/bin/env python3
"""Longest trainable SS2P2 window probe: doubling search over sequence length.

For each (batch, seq) tries one full compute_loss forward+backward on synthetic
events (benchmark config: H=64, L=2, K=62, categorical marks, scan ON) and
reports CUDA peak memory, until OOM. Run on the target GPU node:

    python3 scripts/seq_len_probe.py --batch 1 --batch 64
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from volume_set_mtpp.training.train import create_model

K = 62


def make_batch(B, N, device):
    g = torch.Generator(device="cpu").manual_seed(0)
    dts = torch.rand(B, N, generator=g).mul(0.05).add(1e-3)
    idx = torch.randint(0, K, (B, N), generator=g)
    marks = torch.zeros(B, N, K)
    marks.scatter_(2, idx.unsqueeze(-1), 1.0)
    tmarks = torch.zeros(B, K)
    tmarks.scatter_(1, torch.randint(0, K, (B, 1), generator=g), 1.0)
    return {
        "input_times": dts.to(device),
        "input_marks": marks.to(device),
        "target_time": torch.full((B,), 0.02, device=device),
        "target_marks": tmarks.to(device),
    }


def try_step(model, opt, B, N, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        batch = make_batch(B, N, device)
        opt.zero_grad(set_to_none=True)
        loss, _ = model.compute_loss(batch, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        peak = torch.cuda.max_memory_allocated() / 2**30
        return True, peak
    except torch.cuda.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return False, float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, action="append", default=None)
    ap.add_argument("--start", type=int, default=1024)
    ap.add_argument("--decoder", default="ss2p2")
    args = ap.parse_args()
    batches = args.batch or [1, 8, 64]

    device = torch.device("cuda")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"GPU: {name} ({total:.0f} GiB)")

    config = dict(decoder_type=args.decoder, s2p2_layers=2, s2p2_scan=True,
                  ss2p2_wnorm_cap=6.0, target_rate=38.0,
                  channel_embedding_size=64, time_embedding_size=64,
                  recurrent_hidden_size=64, mark_head="categorical",
                  set_loss_reduction="sum", use_volume=False)
    model = create_model(K, config, device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    print(f"decoder: {type(model.decoder).__name__}; params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for B in batches:
        N, last_ok = args.start, None
        while True:
            ok, peak = try_step(model, opt, B, N, device)
            print(f"  B={B:3d} N={N:>9,}  {'OK  peak %.1f GiB' % peak if ok else 'OOM'}",
                  flush=True)
            if not ok:
                break
            last_ok = (N, peak)
            N *= 2
        # refine between last_ok and the OOM point (two bisection steps)
        if last_ok:
            lo, hi = last_ok[0], N
            for _ in range(2):
                mid = (lo + hi) // 2
                ok, peak = try_step(model, opt, B, mid, device)
                print(f"  B={B:3d} N={mid:>9,}  {'OK  peak %.1f GiB' % peak if ok else 'OOM'}",
                      flush=True)
                if ok:
                    lo = mid
                else:
                    hi = mid
            print(f"== B={B}: longest OK window ~{lo:,} events")


if __name__ == "__main__":
    main()
